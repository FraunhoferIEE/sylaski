from syndatagenerators.data_preparation import datasets 
import cmd_parser
import timecyclegan

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import tqdm_progress
from pytorch_lightning import loggers



if __name__ == "__main__":
    parser = cmd_parser.build_parser()
    parser, cls = timecyclegan.TimeCycleGAN.update_parseargs(parser)

    args: timecyclegan.TCArgs = cmd_parser.run_parser(parser, cls)

    print(str(args))

    tb_logger = loggers.TensorBoardLogger(save_dir=args.log_dir, name=args.experiment, version=args.version)

    time_series_dm = datasets.TStoTSDatamodule(args.fileA, args.fileB, args.batch_size, args.num_workers)

    gan = timecyclegan.TimeCycleGAN(
        time_series_dm.channels,
        128, 
        gan_mode=args.mode,
        batch_size=args.batch_size,
        depth=args.layers,
        residuals=args.residuals,
        dilations=args.dilations,
        d_lr=args.lr_d,
        g_lr=args.lr_g,
        total_epochs=args.epochs, 
        decay_epochs=args.decay_epochs,
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=args.epochs,
        callbacks=[tqdm_progress.TQDMProgressBar(refresh_rate=1)],
        logger=tb_logger,
        log_every_n_steps=1,
    )

    trainer.fit(gan, time_series_dm)
