import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

## Claude modefied
import constraints as constraint_lib
import discriminator as disc_lib
import fhs_constrained
## end of Claude modifications

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  for _ in range(config.sampling.num_sample_batches):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples)
      model.compute_generative_perplexity(text_samples)
  print('Text samples:', text_samples)
  if not config.sampling.semi_ar:
    print('Generative perplexity:',
          model.gen_ppl_metric.compute())
  return text_samples

def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _train_discriminator(config, logger, tokenizer):
  """Train the h_φ discriminator with BCE loss."""
  logger.info('Training discriminator h_phi.')

  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  model.eval()

  constraint = constraint_lib.build_constraint(config, tokenizer, model.mask_index)
  if constraint is None:
    raise ValueError('discriminator_train mode requires a constraint in config.')

  jigsaw_csv = getattr(config.discriminator, 'jigsaw_csv_path', '')
  if jigsaw_csv:
    if not os.path.isabs(jigsaw_csv):
      jigsaw_csv = os.path.join(hydra.utils.get_original_cwd(), jigsaw_csv)
    threshold = getattr(config.constraint, 'threshold', 0.5)
    jigsaw_ds = disc_lib.JigsawDataset(
      csv_path=jigsaw_csv,
      tokenizer=tokenizer,
      max_length=config.model.length,
      threshold=threshold,
    )
    train_ds = torch.utils.data.DataLoader(
      jigsaw_ds,
      batch_size=config.loader.batch_size,
      shuffle=True,
      num_workers=getattr(config.loader, 'num_workers', 4),
      pin_memory=True,
    )
  else:
    train_ds, _ = dataloader.get_dataloaders(config, tokenizer)

  disc = disc_lib.HDiscriminator(
    vocab_size=model.vocab_size,
    hidden_size=getattr(config.discriminator, 'hidden_size', 256),
    num_layers=getattr(config.discriminator, 'num_layers', 4),
    num_heads=getattr(config.discriminator, 'num_heads', 4),
    max_length=getattr(config.discriminator, 'max_length', 1024),
    dropout=getattr(config.discriminator, 'dropout', 0.1),
  )

  save_path = getattr(config.discriminator, 'save_path', 'discriminator.pt')
  if not os.path.isabs(save_path):
    save_path = os.path.join(hydra.utils.get_original_cwd(), save_path)
  disc_lib.train_discriminator(
    discriminator=disc,
    diffusion=model,
    constraint=constraint,
    train_dataloader=train_ds,
    config=config,
    device=model.device,
    save_path=save_path,
  )


def _constrained_sample_eval(config, logger, tokenizer):
  """Generate samples under hard constraint using FHS."""
  logger.info('Constrained FHS sampling.')

  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    model.ema = None

  constraint = constraint_lib.build_constraint(config, tokenizer, model.mask_index)

  # Optionally load a pre-trained discriminator
  disc = None
  disc_path = getattr(config, 'discriminator', None)
  if disc_path is not None:
    disc_path = getattr(config.discriminator, 'load_path', None)
  if disc_path is not None and disc_path != '':
    if not os.path.isabs(disc_path):
      disc_path = os.path.join(hydra.utils.get_original_cwd(), disc_path)
    if not os.path.exists(disc_path):
      logger.warning(
        f'Discriminator file not found: {disc_path}. '
        f'Falling back to analytical constraint only.'
      )
    else:
      disc = disc_lib.load_discriminator(disc_path, config, model.vocab_size)
      disc = disc.to(model.device).eval()
      logger.info(f'Loaded discriminator from {disc_path}')

  topk = getattr(getattr(config, 'discriminator', object()), 'topk', 50)

  model.gen_ppl_metric.reset()
  all_samples = []
  for _ in range(config.sampling.num_sample_batches):
    samples = fhs_constrained.restore_and_sample_constrained(
      diffusion=model,
      constraint=constraint,
      discriminator=disc,
      topk=topk,
    )
    texts = model.tokenizer.batch_decode(samples)
    all_samples.extend(texts)
    model.compute_generative_perplexity(texts)

  print('Constrained samples:')
  for t in all_samples[:config.sampling.num_sample_log]:
    print(' ', t)

  print(f'Generative perplexity: {model.gen_ppl_metric.compute():.3f}')

  if constraint is not None:
    ids = model.tokenizer(all_samples, return_tensors='pt',
                          padding=True, truncation=True,
                          max_length=config.model.length).input_ids
    sat = constraint.check(ids).float().mean().item()
    print(f'Constraint satisfaction rate: {sat:.3f}')


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    generate_samples(config, logger, tokenizer)
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'discriminator_train':
    _train_discriminator(config, logger, tokenizer)
  elif config.mode == 'constrained_sample_eval':
    _constrained_sample_eval(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()