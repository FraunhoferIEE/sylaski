import torch
import torch.nn.functional as F
import copy


class DiffusionUtilities():
    def __init__(self,
                 size: int,
                 network,
                 betaMin: float = 10**-4,
                 betaMax: float = 0.2,
                 n_steps: int = 200,
                 
                 device: str = "cpu"):
        
        """The DiffusionUtilities class provides the operative functions that contain forwards and backwards operations.
        It also contains some precalculated values that are necessary to peform mentioned operations.


        Args:
            network (Neural Network): NN that is designed to predict noise.
            n_steps (int, optional): Number of noising steps. Defaults to 100.
            betaMin (_type_, optional): Smallest beta value for beta schedual. Defaults to 10**-4.
            betaMax (float, optional): Biggest beta value for beta schedual. Defaults to 0.2.
            device (str, optional): The device the training is peformed on. Defaults to "cpu".
        """
        self.network = network.to(device)
        self.size = size
        self.ema_model = copy.deepcopy(self.network).eval().requires_grad_(False)
        self.device = device
        self.betaMin = betaMin
        self.betaMax = betaMax
        self.n_steps = n_steps

        # precalc for diffusion task----------------------------------------------------------------------------------
        self.betas = torch.linspace(betaMin, betaMax, self.n_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.tensor(
            [torch.prod(self.alphas[:i+1])for i in range(len(self.alphas))]).to(device)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            1 - self.alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # ------------------------------------------------------------------------------------------------------------


    def predictNoise(self, x, t, cat,cont):
        """Feeds a image and target timestep t to a network that predicts the noise, that was used
        to create the noised image at step t. Also takes categorical and continuous conditions as input.

        Args:
            image (torch.tensor): Image at timestep t
            t (torch.tensor): Target timestep
            cat (torch.tensor): Categorical conditions
            cont (torch.tensor): Continuous conditions
        Returns:
            torch.tensor : Predicted noise that was used to create image at timestep t
        """

        return self.network(x, t, cat,cont)

    def subtractNoise(self,  x, t, y, cont):
        """Uses trained networkt to predict the noise in every timestep from n_steps till 0. In every step the predicted noise
        is used to perform an operation that returns a less noisy image at timestep t-1. 

        Args:
            image (torch.tensor): Image/sample 
            t (torch.tensor): Target timestemp t

        Returns:
            torch.tensor : Less noisy image/sample at timestemp t-1 using the predicted noise from network.
        """
        t_batch = t.repeat([x.shape[0],1])
        t_batch2 = t.repeat([x.shape[0],x.shape[1],1])
        alpha_t = self.alphas[t_batch2]
        alphas_cp = self.alphas_cumprod[t_batch2]        
        pred = self.predictNoise(x,t_batch.squeeze(),y,cont)
        
        prevImage = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) /(1 - alphas_cp).sqrt() * pred)
   
        if t.item() > 0:
            beta_t = self.betas[t]
            sigma_t = beta_t.sqrt()

            prevImage = prevImage + sigma_t * torch.randn_like(x)
        return prevImage.view(-1,x.shape[1],self.size)

    def addNoise(self, image, t, noise):
        """Adds noise to an image/sample in respect to the provided timestep t.

        Args:
            image (torch.tensor): single/batch of values
            t (torch.tensor): Target timestep 
            noise (torch.tensor): The noise that is added to the image
        Returns:
            torch.tensor : Noised image at timestep t using the noise variable
        """

        view_shape = [1]*len(image.shape)
        view_shape[0] = image.shape[0]
        alphas_cp = self.alphas_cumprod[t]

        return alphas_cp.sqrt().view(view_shape) * image + (1-alphas_cp).sqrt().view(view_shape) * noise