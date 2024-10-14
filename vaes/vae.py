import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VAE(BaseVAE):


    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 fix_points: List = None,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        print("Initialize VAE")

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder_conv = nn.ModuleList()
        self.encoder_batchnorm = nn.ModuleList()
        self.encoder_act = nn.ModuleList()
        
        self.decoder_conv = nn.ModuleList()
        self.decoder_batchnorm = nn.ModuleList()
        self.decoder_act = nn.ModuleList()

        depth = len(hidden_dims)



        # conv = curves.Conv2d
        conv = nn.Conv2d
        # conv_transpose = curves.ConvTranspose2d
        conv_transpose = nn.ConvTranspose2d
        # batchnorm = curves.BatchNorm2d
        batchnorm = nn.BatchNorm2d
        # linear = curves.Linear
        linear = nn.Linear

        # Build Encoder
        # for h_dim in hidden_dims:
        for i in range(depth):
            # modules.append(
            #     nn.Sequential(
            #         curves.Conv2d(in_channels, out_channels=h_dim,
            #                   kernel_size= 3, stride= 2, padding  = 1),
            #         curves.BatchNorm2d(h_dim),
            #         nn.LeakyReLU())
            # )
            # in_channels = h_dim
            self.encoder_conv.append(conv(in_channels, out_channels=hidden_dims[i],
                           kernel_size=3, stride=2, padding=1, 
                        #    fix_points=fix_points
            ))
            in_channels = hidden_dims[i]
            self.encoder_batchnorm.append(batchnorm(hidden_dims[i], 
                                                    # fix_points=fix_points
                                                    ))
            self.encoder_act.append(nn.LeakyReLU())
            

        # self.encoder = nn.Sequential(*modules)
        self.fc_mu = linear(hidden_dims[-1]*4, latent_dim, 
                            # fix_points=fix_points
                            )
        self.fc_var = linear(hidden_dims[-1]*4, latent_dim, 
                            #  fix_points=fix_points
                             )


        # Build Decoder
        modules = []

        self.decoder_input = linear(latent_dim, hidden_dims[-1] * 4, 
                                    # fix_points=fix_points
                                    )

        hidden_dims.reverse()

        # self.decoder_conv = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.decoder_conv.append(conv_transpose(hidden_dims[i],
                           hidden_dims[i + 1],kernel_size= 3,stride= 2,padding= 1,output_padding= 1,
                        #    fix_points=fix_points
            ))
            self.decoder_batchnorm.append(batchnorm(hidden_dims[i + 1], 
                                                    # fix_points=fix_points
                                                    ))
            self.decoder_act.append(nn.LeakyReLU())
            # modules.append(
            #     nn.Sequential(
            #         curves.ConvTranspose2d(hidden_dims[i],
            #                            hidden_dims[i + 1],
            #                            kernel_size=3,
            #                            stride = 2,
            #                            padding=1,
            #                            output_padding=1),
            #         curves.BatchNorm2d(hidden_dims[i + 1]),
            #         nn.LeakyReLU())
            # )




        # self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.ModuleList()
        self.final_layer.append(conv_transpose(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1,
                                            #    fix_points=fix_points
                                               )
                                               )
        self.final_layer.append(batchnorm(hidden_dims[-1], 
                                        #   fix_points=fix_points
                                          ))
        self.final_layer.append(nn.LeakyReLU())
        self.final_layer.append(conv(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1,
                                    #   fix_points= fix_points
                                      ))
        self.final_layer.append(nn.Tanh())

        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, curves.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         for i in range(m.num_bends):
        #             getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
        #             getattr(m, 'bias_%d' % i).data.zero_()


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        for layer, batchnorm, activation in zip(self.encoder_conv, self.encoder_batchnorm ,self.encoder_act):
        # for layer, batchnorm, activation in zip(layers, batchnorms, activations):
            # result = activation(batchnorm(layer(result)))
            input = layer(input)
            input = batchnorm(input)
            input = activation(input)
    

        result = torch.flatten(input, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        # result = self.decoder(result)
        
        for layer, batchnorm, activation in zip(self.decoder_conv, self.decoder_batchnorm ,self.decoder_act):
            # for layer, batchnorm, activation in zip(layers, batchnorms, activations):
            # result = activation(batchnorm(layer(result)))
            result = layer(result)
            result = batchnorm(result)
            result = activation(result)
        
        # result = self.final_layer(result, coeffs_t)
        for layer in self.final_layer:
            # if issubclass(type(layer), curves.CurveModule):
            #     result = layer(result, coeffs_t)
            # else:
            result = layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

        # T^(l)BT^(l-1)
        # mu var 


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]