import time, os, hydra, copy, torch
from DataUtil import Setup

@hydra.main(version_base=None, config_path="./configs", config_name="runBasicCNNVAE_optim")
def main(cfg):
    run(cfg)
    
def run(cfg):
    model = Setup.getModel(cfg)
    dataset = Setup.getDataset(cfg)
    trainer = Setup.getTrainer(cfg, model, dataset)
    logger = Setup.getLogger(cfg, model, trainer)

    early_stopping_count = 0
    early_stopping_logs = cfg.early_stopping_logs
    min_mmd = 1e10

    min_epochs = cfg.min_epochs
    num_epochs = cfg.num_epochs
    print_interval = max(min_epochs // 20, 1)

    s_time = time.time()
    copyModel = None
    for epoch in range(num_epochs):
        loss = trainer.train()
        if(epoch % print_interval == 0):
            mmd = logger.log(loss, epoch, s_time)
            if(torch.isnan(torch.tensor(mmd)).item()):
                print("Abort. Cause: nan-value")
                break
            if(mmd < min_mmd and epoch != 0):
                min_mmd = mmd
                copyModel = copy.deepcopy(model)
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            if(epoch > min_epochs and early_stopping_count >= early_stopping_logs):
                    break
    if(copyModel is not None):
        copyModel.saveModel("", additional_content={"cfg": cfg, "min_mmd": min_mmd})
    print(f"finished training in {epoch} epochs and {time.time() - s_time} seconds. Min mmd: {min_mmd}")
    time.sleep(10)
    logger.lp.terminate_value.value = 1
    logger.lp.join()

if(__name__ == "__main__"):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main()
