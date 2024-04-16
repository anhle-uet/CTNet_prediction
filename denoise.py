import argparse
from pathlib import Path

import numpy as np
import torch
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible
from matplotlib import pyplot as plt

import model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inDir', type=str, default=None, help='Path to input directory')
    parser.add_argument('--outDir', type=str, default=None, help='Path to output directory')
    parser.add_argument('--modelFile', type=str, default='weights/highdose.pth', help='Path to the model file')
    parser.add_argument('--predictMode', type=str, default='full', help='Prediction mode, tile | full')

    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--GPU_id', type=str, default='0',
                        help='Id of GPUs')

    parser.add_argument('--model_name', type=str, default='gtd',
                        help='Choose the type of model to train or test')
    
    parser.add_argument('--patch_size', type=int, default=48,
                        help='output patch size')
    parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
    
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--crop_batch_size', type=int, default=8,
                        help='input batch size for training')
    parser.add_argument('--patch_dim', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--freeze_norm', action='store_true')
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_mlp', action='store_true')
    parser.add_argument('--pos_every', action='store_true')
    parser.add_argument('--no_pos', action='store_true')
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=20000,
                        help='set the max_seq_length of positional embedding')
    
    parser.add_argument('--mode', type=str, default='test',
                    help='Choose to train or test or inference')
    return parser.parse_args()


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def tiled_predict_pad(im, net, ps, overlap, device, max_value=65535.):
    '''
    Tile the image to save GPU memory.
    Process it using our network.
    We use padding for the last patch in each row/column.

    
    Parameters
    ----------
    im: numpy array
        2D image we want to process
    net: a pytorch model
        the network we want to use
    ps: int
        the widht/height of the square tiles we want to use in pixels
    overlap: int
        number of pixels we want the tiles to overlab in x and y
    device:
        The device your network lives on, e.g. your GPU
        
    Returns
    ----------
    est: numpy array
        Denoised image.
    '''
    est = np.zeros(im.shape, dtype=np.uint16)
    xmin = 0
    ymin = 0
    xmax = ps
    ymax = ps
    ovLeft = 0
    while (xmin < im.shape[1]):
        ovTop = 0
        while (ymin < im.shape[0]):

            inputPatch = im[ymin:ymax,xmin:xmax]
            padX = ps-inputPatch.shape[1]
            padY = ps-inputPatch.shape[0]
             
            inputPatch = np.pad(inputPatch,((0, padY),(0,padX)), 'constant', constant_values=(0., 0.) )

            _inputPatch = np.float32(inputPatch / max_value)
            _inputPatch = uint2tensor3(_inputPatch)
            _inputPatch = _inputPatch[None, :]

            with torch.no_grad():
                _inputPatch = _inputPatch.to(device)
                _output = net(_inputPatch)

            _output = _output.cpu()
            _output = _output.squeeze()
            _output = _output.mul(max_value)

            output = np.uint16(_output.detach().clamp(0, max_value).round().numpy())
            output = output[:output.shape[0] - padY, :output.shape[1] - padX]

            est[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = output[ovTop:, ovLeft:]

            ymin=ymin-overlap+ps
            ymax=ymin+ps
            ovTop=overlap//2
        ymin=0
        ymax=ps
        xmin=xmin-overlap+ps
        xmax=xmin+ps
        ovLeft=overlap//2
    return est


def app():
    args = parse_args()

    input_dirpath = Path(args.inDir).resolve()
    if not input_dirpath.exists() or not input_dirpath.is_dir():
        raise ValueError(f'Given inDir is not a directory or does not exist, got {input_dirpath}')
    
    output_dirpath = Path(args.outDir).resolve()
    output_dirpath.mkdir(exist_ok=True, parents=True)

    model_filepath = Path(args.modelFile).resolve()
    if not model_filepath.exists() or not model_filepath.is_file():
        raise ValueError(f'Given modelFile is not a file or does not exist, got {model_filepath}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = model.Model(args, model_filepath.as_posix())

    if args.predictMode == 'tile':
        for img_filepath in input_dirpath.glob('*.tif'):
            img = imread(img_filepath)
            plt.imshow(img, cmap='magma')
            plt.show()

            img_denoised = tiled_predict_pad(img, net, 128, 48, device)
            plt.imshow(img_denoised, cmap='magma')
            plt.show()
    elif args.predictMode == 'full':
        max_value = 65535.
        for img_filepath in input_dirpath.glob('*.tif'):
            img = imread(img_filepath)

            input = np.float32(img / max_value)
            input = uint2tensor3(input)
            input = input[None, :]
            with torch.no_grad():
                input = input.to(device)
                output = net(input)

            output = output.cpu()
            output = output.squeeze()
            output = output.mul(max_value)

            output_np = np.uint16(output.detach().clamp(0, max_value).round().numpy())
            save_tiff_imagej_compatible(Path(output_dirpath, img_filepath.name).as_posix(), output_np, 'YX')


if __name__ == '__main__':
    app()