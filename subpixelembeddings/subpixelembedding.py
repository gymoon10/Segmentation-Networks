import torch, torchvision
import subpixels.networks as networks
import subpixels.log_utils as log_utils
import torch
import torch.nn as nn


class SPiNModel(nn.Module):

    def __init__(self,
                 input_channels,
                 encoder_type_subpixel_embedding,
                 n_filters_encoder_subpixel_embedding,
                 decoder_type_subpixel_embedding,
                 output_channels_subpixel_embedding,
                 n_filter_decoder_subpixel_embedding,
                 output_func_subpixel_embedding,
                 encoder_type_segmentation,
                 n_filters_encoder_segmentation,
                 resolutions_subpixel_guidance,
                 n_filters_subpixel_guidance,
                 n_convolutions_subpixel_guidance,
                 decoder_type_segmentation,
                 n_filters_decoder_segmentation,
                 n_filters_learnable_downsampler,
                 kernel_sizes_learnable_downsampler,
                 weight_initializer,
                 activation_func,
                 use_batch_norm,
                 device=torch.device('cuda')):
        super().__init__()

        self.encoder_type_subpixel_embedding = encoder_type_subpixel_embedding
        self.decoder_type_subpixel_embedding = decoder_type_subpixel_embedding
        self.encoder_type_segmentation = encoder_type_segmentation
        self.decoder_type_segmentation = decoder_type_segmentation
        self.resolutions_subpixel_guidance = resolutions_subpixel_guidance
        self.device = device

        assert len(resolutions_subpixel_guidance) < len(n_filters_decoder_segmentation)

        '''
        Build subpixel embedding, can be replaced with hand-crafted upsampling (bilinear, nearest)
        '''
        self.use_interpolated_upsampling = \
            'upsample' in encoder_type_subpixel_embedding or \
            'upsample' in decoder_type_subpixel_embedding

        self.use_bilinear_upsampling = \
            self.use_interpolated_upsampling and \
            'bilinear' in self.encoder_type_subpixel_embedding and \
            'bilinear' in self.decoder_type_subpixel_embedding

        self.use_nearest_upsampling = \
            self.use_interpolated_upsampling and \
            'nearest' in self.encoder_type_subpixel_embedding and \
            'nearest' in self.decoder_type_subpixel_embedding

        self.use_subpixel_guidance = \
            'none' not in encoder_type_subpixel_embedding and \
            'none' not in decoder_type_subpixel_embedding and \
            not self.use_interpolated_upsampling

        # Select subpixel embedding encoder
        if 'resnet5_subpixel_embedding' in encoder_type_subpixel_embedding:
            self.encoder_subpixel_embedding = networks.SubpixelEmbeddingEncoder(
                n_layer=5,
                input_channels=input_channels,
                n_filters=n_filters_encoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet7_subpixel_embedding' in encoder_type_subpixel_embedding:
            self.encoder_subpixel_embedding = networks.SubpixelEmbeddingEncoder(
                n_layer=7,
                input_channels=input_channels,
                n_filters=n_filters_encoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'none' in encoder_type_subpixel_embedding or self.use_interpolated_upsampling:
            self.encoder_subpixel_embedding = None
        else:
            raise ValueError('Unsupported encoder type {}'.format(encoder_type_subpixel_embedding))

        # Latent channels is number channels in the last layer of encoder
        latent_channels_subpixel_embedding = n_filters_encoder_subpixel_embedding[-1]

        # Select subpixel embedding decoder
        if 'subpixel' in decoder_type_subpixel_embedding:
            self.decoder_subpixel_embedding = networks.SubpixelEmbeddingDecoder(
                input_channels=latent_channels_subpixel_embedding,
                output_channels=output_channels_subpixel_embedding,
                scale=2,
                n_filter=n_filter_decoder_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func=output_func_subpixel_embedding,
                use_batch_norm=use_batch_norm)
        elif 'none' in decoder_type_subpixel_embedding or self.use_interpolated_upsampling:
            self.decoder_subpixel_embedding = None
        else:
            raise ValueError('Unsupported decoder type: {}'.format(decoder_type_subpixel_embedding))

        '''
        Build segmentation network
        '''
        # Select segmentation encoder
        if 'resnet18' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=18,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet34' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=34,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        elif 'resnet50' in encoder_type_segmentation:
            self.encoder_segmentation = networks.ResNetEncoder(
                n_layer=50,
                input_channels=input_channels,
                n_filters=n_filters_encoder_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Unsupported segmentation encoder type: {}'.format(
                encoder_type_segmentation))

        # Build skip connections based on output channels in segmentation encoder
        # n_filters_encoder_segmentation up to last one (omit latent)
        # the prepended 0's are to account for original and 2x resolution skips
        skip_channels_segmentation = [0, 0] + n_filters_encoder_segmentation[:-1]

        if self.use_interpolated_upsampling:
            skip_channels_segmentation[0] = 1

        latent_channels_segmentation = n_filters_encoder_segmentation[-1]

        '''
        Build Subpixel Guidance Modules

        0 -> scale = 1 -> 2x resolution (output size of SubpixelGuidanceDecoder)
        1 -> scale = 2-> 1x resolution
        '''
        if self.use_subpixel_guidance:

            for resolution in resolutions_subpixel_guidance:
                assert resolution <= 5, 'Unsupported resolution for subpixel guidance: {}'

            # Ensure n_filters_subpixel_guidance is compatible with SPG module.
            assert isinstance(resolutions_subpixel_guidance, list) and \
                isinstance(n_filters_subpixel_guidance, list) and \
                isinstance(n_convolutions_subpixel_guidance, list), \
                'Arguments (resolutions, n_filters, n_convolutions) for subpixel guidance must be lists'

            assert len(resolutions_subpixel_guidance) == len(n_filters_subpixel_guidance) and \
                len(resolutions_subpixel_guidance) == len(n_convolutions_subpixel_guidance), \
                'Arguments (resolutions, n_filters, n_convolutions) must have same length'

            scales = [
                int(2 ** s) for s in resolutions_subpixel_guidance
            ]

            self.subpixel_guidance = networks.SubpixelGuidance(
                scales=scales,
                n_filters=n_filters_subpixel_guidance,
                n_convolutions=n_convolutions_subpixel_guidance,
                subpixel_embedding_channels=output_channels_subpixel_embedding,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            # Add skip connections from subpixel embedding starting from the end
            for (idx, n_filters) in zip(resolutions_subpixel_guidance, n_filters_subpixel_guidance):
                # Have a convolution at that resolution
                if n_filters > 0:
                    skip_channels_segmentation[idx] += n_filters
                # No convolution at that resolution -> use # channels from space to depth output
                else:
                    skip_channels_segmentation[idx] += int(4 ** idx * output_channels_subpixel_embedding)

        # If we don't plan on building subpixel guidance
        else:
            self.subpixel_guidance = None

        # Reverse list to build layers for decoder
        skip_channels_segmentation = skip_channels_segmentation[::-1]

        # Segmentation Decoder
        if 'subpixel_guidance' in decoder_type_segmentation:

            if 'learnable_downsampler' not in decoder_type_segmentation:
                n_filters_learnable_downsampler = []
                kernel_sizes_learnable_downsampler = []

            self.decoder_segmentation = networks.SubpixelGuidanceDecoder(
                input_channels=latent_channels_segmentation,
                output_channels=3,
                n_filters=n_filters_decoder_segmentation,
                n_skips=skip_channels_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm=use_batch_norm,
                n_filters_learnable_downsampler=n_filters_learnable_downsampler,
                kernel_sizes_learnable_downsampler=kernel_sizes_learnable_downsampler)
        elif 'generic' in decoder_type_segmentation:
            self.decoder_segmentation = networks.GenericDecoder(
                input_channels=latent_channels_segmentation,
                output_channels=3,
                n_filters=n_filters_decoder_segmentation,
                n_skips=skip_channels_segmentation,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm=use_batch_norm,
                full_resolution_output='1x' in decoder_type_segmentation)
        else:
            raise ValueError('Unsupported segmentation decoder type: {}'.format(
                decoder_type_segmentation))

        # Ensure that modules are removed if not using subpixel guidance
        if not self.use_subpixel_guidance:
            assert self.encoder_subpixel_embedding is None and \
                self.decoder_subpixel_embedding is None and \
                self.subpixel_guidance is None, \
                'Subpixel encoder and decoder types must be none if not using subpixel guidance:\n' + \
                'encoder_subpixel_embedding={}\n' + \
                'decoder_subpixel_embedding={}'.format(
                    encoder_type_subpixel_embedding, decoder_type_subpixel_embedding)

        # Move to device
        # self.to(self.device)

    def forward(self, input_scan):
        '''
        Forwards the input through the network

        Arg(s):
            input_scan : torch.Tensor[float32]
                input MRI scan

        Returns:
            list[torch.Tensor] : lesion segmentation in a list
        '''

        # Remove extra dimension (should be 1) from N x C x D x H x W to get N x C x H x W
        if len(input_scan.shape) == 5:
            input_scan = input_scan[:, :, 0, :, :]
        self.input_scan = input_scan

        if self.use_subpixel_guidance:
            # Forward through subpixel embedding to get N x M x 2H x 2W
            latent_subpixel_embedding, \
                skips_subpixel_embedding = self.encoder_subpixel_embedding(input_scan)

            output_subpixel_embedding = \
                self.decoder_subpixel_embedding(latent_subpixel_embedding)
            self.output_subpixel_embedding = output_subpixel_embedding[-1]

        # Forward original tensor through segmentation encoder
        latent_segmentation, skips_segmentation = self.encoder_segmentation(input_scan)

        # Add placeholder for 2x and 1x resolution skips
        skips_segmentation = [None, None] + skips_segmentation

        # Use upsampling instead of subpixel guidance
        if self.use_interpolated_upsampling:

            if self.use_bilinear_upsampling:
                interpolated_scan = torch.nn.functional.interpolate(
                    input_scan,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)
            elif self.use_nearest_upsampling:
                interpolated_scan = torch.nn.functional.interpolate(
                    input_scan,
                    scale_factor=2,
                    mode='nearest')
            else:
                raise ValueError('Must specify bilinear or nearest interpolation type.')
            skips_segmentation[0] = interpolated_scan

        '''
        Forward through Subpixel Guidance
        '''
        if self.use_subpixel_guidance:
            # Calculate desired output shapes for each skip connection in subpixel guidance
            skips_segmentation_shapes = [
                skip.shape[-2:] for skip in skips_segmentation if skip is not None
            ]
            skips_segmentation_shapes = \
                [self.output_subpixel_embedding.shape[-2:], input_scan.shape[-2:]] + skips_segmentation_shapes
            skips_segmentation_shapes = [
                skips_segmentation_shapes[resolution] for resolution in self.resolutions_subpixel_guidance
            ]

            # Space to Depth to build skip connections
            skips_subpixel_guidance = self.subpixel_guidance(
                self.output_subpixel_embedding,
                skips_segmentation_shapes)

            # Concatenate segmentation encoder skips with remaining subpixel guidance skips
            for (resolution, skip_subpixel_guidance) in zip(self.resolutions_subpixel_guidance, skips_subpixel_guidance):
                if skips_segmentation[resolution] is None:
                    skips_segmentation[resolution] = skip_subpixel_guidance
                else:
                    skips_segmentation[resolution] = torch.cat(
                        [skips_segmentation[resolution], skip_subpixel_guidance],
                        dim=1)

        # Forward through decoder & return output
        logits = self.decoder_segmentation(latent_segmentation, skips_segmentation)

        return logits

class SPIN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.segment = SPiNModel(input_channels=3,
                 encoder_type_subpixel_embedding='resnet5_subpixel_embedding',
                 n_filters_encoder_subpixel_embedding=[16, 16, 16],
                 decoder_type_subpixel_embedding='subpixel',
                 output_channels_subpixel_embedding=8,
                 n_filter_decoder_subpixel_embedding=16,
                 output_func_subpixel_embedding='linear',
                 encoder_type_segmentation='resnet18',
                 n_filters_encoder_segmentation=[32, 64, 128, 196, 196],
                 resolutions_subpixel_guidance=[0, 1],
                 n_filters_subpixel_guidance=[8, 8],
                 n_convolutions_subpixel_guidance=[1, 1],
                 decoder_type_segmentation='generic',
                 n_filters_decoder_segmentation=[196, 128, 64, 32, 16],
                 n_filters_learnable_downsampler=[16, 16],
                 kernel_sizes_learnable_downsampler=[3, 3],
                 weight_initializer='kaiming_normal',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3,
                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        )

    def forward(self, image):
        out = self.segment(image)
        out = self.up(out[0])

        return out