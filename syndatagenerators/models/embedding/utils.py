import matplotlib.pyplot as plt

def plot_test_samples(test_samples):
    for batch_id in range(8):
        plt.subplot(2, 4, batch_id+1)
        plt.title(test_samples['x_cat'][batch_id, 0, 0].item())
        plt.plot(test_samples['y'][batch_id, :, 0].cpu().numpy(), label='y')
        plt.plot(test_samples['pred'][batch_id, :, 0].detach().cpu().numpy(), label='pred')
    return

def filter_by_task_id(test_samples, task_id):
    mask = test_samples['x_cat'][:, 0, 0]==task_id
    filtered_samples = {}
    filtered_samples['x_cat'] = test_samples['x_cat'][mask]
    filtered_samples['x_cont'] = test_samples['x_cont'][mask]
    filtered_samples['y'] = test_samples['y'][mask]
    filtered_samples['pred'] = test_samples['pred'][mask]
    return filtered_samples