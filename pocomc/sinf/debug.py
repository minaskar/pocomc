import shutil
import pathlib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple
from natsort import natsort

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
import matplotlib.transforms as matplotlib_transforms
from matplotlib.animation import FuncAnimation

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Tag:
    train: str = 'train'
    validation: str = 'validation'
    generative: str = 'generative'


class Mode:
    forward: str = 'forward'
    inverse: str = 'inverse'


@dataclass
class FileWriterConfig:
    # What kind of data to write
    write_training_paths: bool = False
    write_validation_paths: bool = False
    write_generative_paths: bool = False
    write_training_reconstructions: bool = False
    write_validation_reconstructions: bool = False
    write_scalars: bool = False
    write_samples: bool = False

    @property
    def enabled(self):
        return any([
            self.write_training_paths,
            self.write_validation_paths,
            self.write_generative_paths,
            self.write_training_reconstructions,
            self.write_validation_reconstructions,
            self.write_scalars
        ])


@dataclass
class VisualizerConfig:
    # What to plot
    plot_training_paths: bool = False
    plot_validation_paths: bool = False
    plot_generative_paths: bool = False
    plot_training_reconstructions: bool = False
    plot_validation_reconstructions: bool = False
    plot_scalars: bool = True

    # Where to render/display plots
    tensorboard: bool = False  # Show plots in tensorboard
    show_figures: bool = False  # Show plots in the matplotlib viewer
    save_figures: bool = False  # Save plots as files

    # What to animate
    animate_training_paths: bool = False
    animate_validation_paths: bool = False
    animate_generative_paths: bool = False
    animate_training_reconstructions: bool = False
    animate_validation_reconstructions: bool = False

    # Dimensions to plot
    dim0: int = 0
    dim1: int = 1

    # Colors and such for various kinds of scenarios
    path_color: str = 'gray'
    path_alpha: float = 0.3

    data_space_color: str = 'tab:orange'
    data_space_marker: str = 'o'
    data_space_alpha: float = 0.3
    data_space_label: str = 'Data points'

    latent_space_color: str = 'tab:green'
    latent_space_marker: str = '^'
    latent_space_alpha: float = 0.3
    latent_contour_alpha: float = 0.3
    latent_space_label: str = 'Latent points'

    plot_latent_contours: bool = True
    latent_contour_label: str = 'Latent Gaussian'
    latent_contour_line_width: float = 2.0
    n_latent_points: int = 100

    reconstruction_color: str = 'tab:blue'
    reconstruction_marker: str = '^'
    reconstruction_alpha: float = 0.3
    reconstruction_label: str = 'Reconstructed points'

    step_text_location = (0.5, 0.85)

    # Figure-level parameters
    figsize: Optional[Tuple[float, float]] = None
    legend_loc: Optional[str] = 'upper left'

    # Animation parameters
    ffmpeg_path: str = '/usr/bin/ffmpeg'  # On Linux: try `which ffmpeg` in the terminal. Exporting as gif is possible
    # in any case, but mp4 is only possible with ffmpeg or a suitable replacement. GIFs have no compression, so file
    # sizes are quite big (tens to hundreds of megabytes) for long animations.

    animation_interval: int = 10
    animation_extension: str = 'gif'
    animation_dpi: Optional[int] = None

    @property
    def enabled(self):
        return any([
            self.plot_training_paths,
            self.plot_validation_paths,
            self.plot_generative_paths,
            self.plot_training_reconstructions,
            self.plot_validation_reconstructions,
            self.plot_scalars
        ])


class FlowVisualizer:
    def __init__(self, directory: pathlib.Path, config: Optional[VisualizerConfig] = VisualizerConfig()):
        """
        Tool to visualize SINF performance.
        """
        self.directory = directory
        self.paths_directory = self.directory / 'paths'
        self.paths_train_directory = self.paths_directory / Tag.train
        self.paths_validation_directory = self.paths_directory / Tag.validation
        self.paths_generative_directory = self.paths_directory / Tag.generative

        self.reconstructions_directory = self.directory / 'reconstructions'
        self.reconstructions_train_directory = self.reconstructions_directory / Tag.train
        self.reconstructions_validation_directory = self.reconstructions_directory / Tag.validation

        self.paths_train_directory.mkdir(exist_ok=True, parents=True)
        self.paths_validation_directory.mkdir(exist_ok=True, parents=True)
        self.paths_generative_directory.mkdir(exist_ok=True, parents=True)
        self.reconstructions_train_directory.mkdir(exist_ok=True, parents=True)
        self.reconstructions_validation_directory.mkdir(exist_ok=True, parents=True)

        self.config = config
        self.writer = None if not self.config.tensorboard else SummaryWriter(log_dir=str(self.directory))

    @staticmethod
    @torch.no_grad()
    def draw_confidence_ellipse(x: torch.Tensor, y: torch.Tensor, ax, sigma: float = 3.0, **kwargs):
        """
        Create a plot of the covariance confidence ellipse of x and y.

        :param x: array-like, shape (n, ), input data.
        :param y: array-like, shape (n, ), input data.
        :param ax: matplotlib axis object for plotting.
        :param sigma: float, the number of standard deviations to determine the ellipse's radii.
        :param kwargs: forwarded to matplotlib.patches.Ellipse.
        """

        # x = x.cpu().numpy()
        # y = y.cpu().numpy()
        # 
        # cov = np.cov(x, y)
        # pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        pearson = 0
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor='none',
            **kwargs
        )

        # scale_x = np.sqrt(cov[0, 0]) * sigma
        scale_x = sigma
        # mean_x = np.mean(x)
        mean_x = 0

        # scale_y = np.sqrt(cov[1, 1]) * sigma
        scale_y = sigma
        # mean_y = np.mean(y)
        mean_y = 0

        transform = matplotlib_transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transform + ax.transData)
        ax.add_patch(ellipse)

    def draw_latent_ellipse(self, ax):
        if self.config.plot_latent_contours:
            ellipse_x = torch.randn(1000)
            ellipse_y = torch.randn(1000)

            self.draw_confidence_ellipse(
                ellipse_x, ellipse_y, ax=ax, sigma=1,
                edgecolor=self.config.latent_space_color,
                linewidth=self.config.latent_contour_line_width,
                alpha=self.config.latent_contour_alpha,
                label=self.config.latent_contour_label
            )
            self.draw_confidence_ellipse(
                ellipse_x, ellipse_y, ax=ax, sigma=2,
                edgecolor=self.config.latent_space_color,
                linewidth=self.config.latent_contour_line_width,
                alpha=self.config.latent_contour_alpha
            )

    def save_paths_figure(self, fig, tag, mode, global_step):
        if not self.config.save_figures:
            return

        if mode == Mode.forward:
            if tag == Tag.train:
                out_dir = self.paths_train_directory
            elif tag == Tag.validation:
                out_dir = self.paths_validation_directory
            else:
                raise ValueError(f"Dataset type {tag} is not supported")
        elif mode == Mode.inverse:  # TODO make this check rely on Tag.generative instead
            out_dir = self.paths_generative_directory
        else:
            raise ValueError
        fig.savefig(
            out_dir / f'{global_step}-{tag}-({self.config.dim0} {self.config.dim1}).png',
            bbox_inches='tight'
        )

    def add_tensorboard_paths_figure(self, fig, tag, mode, global_step):
        if self.config.tensorboard:
            self.writer.add_figure(
                tag=f'paths/{mode}/{tag}/({self.config.dim0}, {self.config.dim1})',
                figure=fig,
                global_step=global_step,
                close=False
            )

    @torch.no_grad()
    def plot_paths(self, paths: torch.Tensor, tag: str, mode: str, global_step: int):
        """
        Plot paths given by SINF transformations. The plot is in 2D.

        :param tag: string that represents the current plot. If tensorboard is enabled, the constructed figure will be
        shown with this tag. If figure saving is enabled, this tag will be used in the filename. Should be one of
        ['train', 'validation'].
        :param global_step: integer that represents the step of the algorithm. If tensorboard is enabled, the
            constructed figure will be shown with this step value. If figure saving is enabled, this value will be used
            in the filename.
        :param tag: one of 'train', 'validation'
        :param mode: one of 'forward', 'inverse'
        """
        if not any([
            self.config.plot_training_paths,
            self.config.plot_validation_paths,
            self.config.plot_generative_paths
        ]):
            return

        if mode == Mode.forward:
            data_space_index = 0
            latent_space_index = -1
        elif mode == Mode.inverse:
            data_space_index = -1
            latent_space_index = 0
        else:
            raise ValueError(f"Mode {mode} not supported")

        fig, ax = plt.subplots(figsize=self.config.figsize)

        ax.plot(
            paths[..., self.config.dim0],
            paths[..., self.config.dim1],
            color=self.config.path_color,
            alpha=self.config.path_alpha
        )
        ax.plot(
            paths[data_space_index, :, self.config.dim0],
            paths[data_space_index, :, self.config.dim1],
            color=self.config.data_space_color,
            linewidth=0,
            marker=self.config.data_space_marker,
            alpha=self.config.data_space_alpha,
            label=self.config.data_space_label
        )
        ax.plot(
            paths[latent_space_index, :, self.config.dim0],
            paths[latent_space_index, :, self.config.dim1],
            color=self.config.latent_space_color,
            linewidth=0,
            marker=self.config.latent_space_marker,
            alpha=self.config.latent_space_alpha,
            label=self.config.latent_space_label
        )

        self.draw_latent_ellipse(ax)

        ax.set_title(f'Step {global_step} | {tag} | Dim ({self.config.dim0}, {self.config.dim1}) | {mode}')
        ax.legend(loc=self.config.legend_loc)
        fig.tight_layout()

        self.save_paths_figure(fig, tag, mode, global_step)
        self.add_tensorboard_paths_figure(fig, tag, mode, global_step)
        if self.config.show_figures:
            plt.show()
        plt.close(fig)

    @torch.no_grad()
    def plot_forward_paths(self, paths: torch.Tensor, tag: str, global_step: int):
        self.plot_paths(paths=paths, tag=tag, mode=Mode.forward, global_step=global_step)

    @torch.no_grad()
    def plot_inverse_paths(self, paths: torch.Tensor, tag: str, global_step: int):
        self.plot_paths(paths=paths, tag=tag, mode=Mode.inverse, global_step=global_step)

    def save_reconstruction_figure(self, fig, tag, global_step):
        if self.config.save_figures:
            if tag == Tag.train:
                out_dir = self.reconstructions_train_directory
            elif tag == Tag.validation:
                out_dir = self.reconstructions_validation_directory
            else:
                raise ValueError(f"Dataset type {tag} is not supported")
            fig.savefig(
                out_dir / f'{global_step}-{tag}-({self.config.dim0} {self.config.dim1}).png',
                bbox_inches='tight'
            )

    def add_tensorboard_reconstructions_figure(self, fig, tag, global_step):
        if self.config.tensorboard:
            self.writer.add_figure(
                tag=f'reconstructions/{tag}/({self.config.dim0}, {self.config.dim1})',
                figure=fig,
                global_step=global_step,
                close=False
            )

    @torch.no_grad()
    def plot_reconstructions(self,
                             original: torch.Tensor,
                             reconstructed: torch.Tensor,
                             tag: str,
                             global_step: int):
        """
        Plot data points and their reconstructions. Data points are taken to latent space via forward flow
        transformations and back via inverse flow transformations. The plot is 2D.

        :param tag: string that represents the current plot. If tensorboard is enabled, the constructed figure will be
            shown with this tag. If figure saving is enabled, this tag will be used in the filename.
        :param global_step: integer that represents the step of the algorithm. If tensorboard is enabled, the
            constructed figure will be shown with this step value. If figure saving is enabled, this value will be used in
            the filename.
        """
        if not any([
            self.config.plot_training_reconstructions,
            self.config.plot_validation_reconstructions
        ]):
            return

        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.plot(
            original[..., self.config.dim0],
            original[..., self.config.dim1],
            color=self.config.data_space_color,
            linewidth=0,
            marker=self.config.data_space_marker,
            alpha=self.config.data_space_alpha,
            label=self.config.data_space_label
        )
        ax.plot(
            reconstructed[..., self.config.dim0],
            reconstructed[..., self.config.dim1],
            color=self.config.reconstruction_color,
            linewidth=0,
            marker=self.config.reconstruction_marker,
            alpha=self.config.reconstruction_alpha,
            label=self.config.reconstruction_label
        )

        lc = LineCollection(
            [
                (
                    (original[i, self.config.dim0], original[i, self.config.dim1]),
                    (reconstructed[i, self.config.dim0], reconstructed[i, self.config.dim1])
                ) for i in range(len(original))
            ],
            edgecolors=self.config.path_color,
            alpha=self.config.path_alpha
        )
        ax.add_collection(lc)
        ax.set_title(
            f'Step {global_step} | {tag} | dimensions ({self.config.dim0}, {self.config.dim1}) | reconstruction')
        ax.legend(loc=self.config.legend_loc)
        fig.tight_layout()

        self.save_reconstruction_figure(fig, tag, global_step)
        self.add_tensorboard_reconstructions_figure(fig, tag, global_step)
        if self.config.show_figures:
            plt.show()
        plt.close(fig)

    def animate_paths(self, directory: pathlib.Path, mode: str, output_directory: pathlib.Path):
        assert mode in [Mode.forward, Mode.inverse]

        if not any([
            self.config.animate_training_paths,
            self.config.animate_validation_paths,
            self.config.animate_generative_paths
        ]):
            return

        print(f'Animating data in {str(directory)}')
        output_path = output_directory / f'animation-{self.config.dim0}-{self.config.dim1}.{self.config.animation_extension}'
        output_path = output_path.absolute()
        numpy_files = list(natsort.natsorted(directory.glob('*.npy')))
        if len(numpy_files) == 0:
            print(f'No numpy files found in {directory.absolute()}')
            return

        if mode == Mode.forward:
            data_space_index = 0
            latent_space_index = -1
        elif mode == Mode.inverse:
            data_space_index = -1
            latent_space_index = 0
        else:
            raise ValueError(f"Mode {mode} not supported")

        fig, ax = plt.subplots(figsize=self.config.figsize)

        self.draw_latent_ellipse(ax)

        data_space_plot, = ax.plot(
            [],
            [],
            color=self.config.data_space_color,
            linewidth=0,
            marker=self.config.data_space_marker,
            alpha=self.config.data_space_alpha,
            label=self.config.data_space_label
        )
        latent_space_plot, = ax.plot(
            [],
            [],
            color=self.config.latent_space_color,
            linewidth=0,
            marker=self.config.latent_space_marker,
            alpha=self.config.latent_space_alpha,
            label=self.config.latent_space_label
        )
        ax.legend(loc=self.config.legend_loc)

        line_plots = []
        for _ in range(np.load(numpy_files[0]).shape[1]):
            lines_plot, = ax.plot(
                [],
                [],
                color=self.config.path_color,
                alpha=self.config.path_alpha
            )
            line_plots.append(lines_plot)

        step_text = ax.text(
            *self.config.step_text_location, "", bbox=dict(facecolor='w', alpha=0.5, pad=5),
            transform=ax.transAxes, ha="center"
        )

        min_x = np.infty
        min_y = np.infty
        max_x = -np.infty
        max_y = -np.infty


        def update(numpy_file_index):
            numpy_file = numpy_files[numpy_file_index]
            nonlocal min_x, max_x, min_y, max_y
            data = np.load(numpy_file)

            min_x = min(min_x, np.min(data[..., self.config.dim0]))
            min_y = min(min_y, np.min(data[..., self.config.dim1]))
            max_x = max(max_x, np.max(data[..., self.config.dim0]))
            max_y = max(max_y, np.max(data[..., self.config.dim1]))

            dx = (max_x - min_x) * 0.1
            dy = (max_y - min_y) * 0.1
            ax.axis([min_x - dx, max_x + dx, min_y - dy, max_y + dy])

            data_space_plot.set_data(data[data_space_index, :, self.config.dim0],
                                     data[data_space_index, :, self.config.dim1])
            latent_space_plot.set_data(data[latent_space_index, :, self.config.dim0],
                                       data[latent_space_index, :, self.config.dim1])

            for i, line_plot in enumerate(line_plots):
                line_plot.set_data(data[:, i, self.config.dim0], data[:, i, self.config.dim1])

            step_text.set_text(f'Step {numpy_file_index:>5}')

            return data_space_plot, latent_space_plot, step_text, *line_plots

        ani = FuncAnimation(
            fig,
            update,
            interval=self.config.animation_interval,
            blit=True,
            repeat=True,
            frames=list(range(len(numpy_files)))
        )
        ani.save(str(output_path), writer='ffmpeg', dpi=self.config.animation_dpi)
        plt.close(fig)

        print(f'> Output file: {str(output_path)}')

    def animate_forward_paths(self, directory: pathlib.Path, output_directory: pathlib.Path):
        self.animate_paths(directory=directory, mode=Mode.forward, output_directory=output_directory)

    def animate_inverse_paths(self, directory: pathlib.Path, output_directory: pathlib.Path):
        self.animate_paths(directory=directory, mode=Mode.inverse, output_directory=output_directory)

    def animate_reconstructions(self, directory: pathlib.Path, output_directory: pathlib.Path):
        if not any([
            self.config.animate_training_reconstructions,
            self.config.animate_validation_reconstructions
        ]):
            return

        print(f'Animating data in {str(directory)}')
        output_path = output_directory / f'animation-{self.config.dim0}-{self.config.dim1}.{self.config.animation_extension}'
        output_path = output_path.absolute()
        numpy_files = list(natsort.natsorted(directory.glob('*.npy')))
        # Files have arrays with shape (2, n_samples, n_dimensions). The first dimension are original data points,
        # the second are their reconstructions.
        if len(numpy_files) == 0:
            print(f'No numpy files found in {directory.absolute()}')
            return

        fig, ax = plt.subplots(figsize=self.config.figsize)

        original_data_plot, = ax.plot(
            [],
            [],
            color=self.config.data_space_color,
            linewidth=0,
            marker=self.config.data_space_marker,
            alpha=self.config.data_space_alpha,
            label=self.config.data_space_label
        )
        reconstructed_data_plot, = ax.plot(
            [],
            [],
            color=self.config.reconstruction_color,
            linewidth=0,
            marker=self.config.reconstruction_marker,
            alpha=self.config.reconstruction_alpha,
            label=self.config.reconstruction_label
        )
        ax.legend(loc=self.config.legend_loc)

        line_plots = []
        n_samples = np.load(numpy_files[0]).shape[1]
        for _ in range(n_samples):
            lines_plot, = ax.plot(
                [],
                [],
                color=self.config.path_color,
                alpha=self.config.path_alpha
            )
            line_plots.append(lines_plot)

        step_text = ax.text(
            *self.config.step_text_location, "", bbox=dict(facecolor='w', alpha=0.5, pad=5),
            transform=ax.transAxes, ha="center"
        )

        min_x = np.infty
        min_y = np.infty
        max_x = -np.infty
        max_y = -np.infty

        def update(numpy_file_index):
            numpy_file = numpy_files[numpy_file_index]
            nonlocal min_x, max_x, min_y, max_y
            data = np.load(numpy_file)

            min_x = min(min_x, np.min(data[..., self.config.dim0]))
            min_y = min(min_y, np.min(data[..., self.config.dim1]))
            max_x = max(max_x, np.max(data[..., self.config.dim0]))
            max_y = max(max_y, np.max(data[..., self.config.dim1]))

            dx = (max_x - min_x) * 0.1
            dy = (max_y - min_y) * 0.1
            ax.axis([min_x - dx, max_x + dx, min_y - dy, max_y + dy])

            original_data_plot.set_data(data[0, :, self.config.dim0], data[0, :, self.config.dim1])
            reconstructed_data_plot.set_data(data[1, :, self.config.dim0], data[1, :, self.config.dim1])

            for i, line_plot in enumerate(line_plots):
                line_plot.set_data(data[:, i, self.config.dim0], data[:, i, self.config.dim1])

            step_text.set_text(f'Step {numpy_file_index:>5}')

            return original_data_plot, reconstructed_data_plot, step_text, *line_plots

        ani = FuncAnimation(
            fig,
            update,
            interval=self.config.animation_interval,
            blit=True,
            repeat=True,
            frames=list(range(len(numpy_files)))
        )
        ani.save(str(output_path), writer='ffmpeg', dpi=self.config.animation_dpi)
        plt.close(fig)

        print(f'> Output file: {str(output_path)}')

    def add_scalar(self, tag: str, value: float, step: int):
        if self.config.tensorboard:
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)


class FlowFileWriter:
    def __init__(self, directory: pathlib.Path, config: Optional[FileWriterConfig] = FileWriterConfig()):
        self.directory = directory
        self.samples_directory = directory / 'samples'
        self.paths_directory = directory / 'paths'
        self.paths_train_directory = self.paths_directory / Tag.train
        self.paths_validation_directory = self.paths_directory / Tag.validation
        self.paths_generative_directory = self.paths_directory / Tag.generative

        self.reconstructions_directory = directory / 'reconstructions'
        self.reconstructions_train_directory = self.reconstructions_directory / Tag.train
        self.reconstructions_validation_directory = self.reconstructions_directory / Tag.validation

        self.samples_directory.mkdir(exist_ok=True, parents=True)
        self.paths_train_directory.mkdir(exist_ok=True, parents=True)
        self.paths_validation_directory.mkdir(exist_ok=True, parents=True)
        self.paths_generative_directory.mkdir(exist_ok=True, parents=True)
        self.reconstructions_train_directory.mkdir(exist_ok=True, parents=True)
        self.reconstructions_validation_directory.mkdir(exist_ok=True, parents=True)

        self.config = config
        self.scalars = []

    def write_scalar(self, tag: str, value: float, step: int):
        if self.config.write_scalars:
            self.scalars.append({
                'tag': tag,
                'value': float(value),
                'step': step
            })

    def write_training_paths(self, paths: torch.Tensor, step: int):
        if self.config.write_training_paths:
            np.save(file=str(self.paths_train_directory / f'{step:06d}'), arr=paths.detach().cpu().numpy())

    def write_validation_paths(self, paths: torch.Tensor, step: int):
        if self.config.write_validation_paths:
            np.save(file=str(self.paths_validation_directory / f'{step:06d}'), arr=paths.detach().cpu().numpy())

    def write_generative_paths(self, paths: torch.Tensor, step: int):
        if self.config.write_generative_paths:
            np.save(file=str(self.paths_generative_directory / f'{step:06d}'), arr=paths.detach().cpu().numpy())

    def write_samples(self, samples: torch.Tensor, step: int):
        if self.config.write_samples:
            np.save(file=str(self.samples_directory / f'{step:06d}'), arr=samples.detach().cpu().numpy())

    def write_training_reconstructions(self, original: torch.Tensor, reconstructed: torch.Tensor, step: int):
        if self.config.write_training_reconstructions:
            np.save(
                file=str(self.reconstructions_train_directory / f'{step:06d}'),
                arr=np.stack([
                    original.detach().cpu().numpy(),
                    reconstructed.detach().cpu().numpy()
                ])
            )

    def write_validation_reconstructions(self, original: torch.Tensor, reconstructed: torch.Tensor, step: int):
        if self.config.write_validation_reconstructions:
            np.save(
                file=str(self.reconstructions_validation_directory / f'{step:06d}'),
                arr=np.stack([
                    original.detach().cpu().numpy(),
                    reconstructed.detach().cpu().numpy()
                ])
            )


class FlowDebugger:
    def __init__(self,
                 directory: Optional[pathlib.Path] = None,
                 directory_suffix: Optional[str] = None,
                 delete_existing: bool = False,
                 file_writer_config: Optional[FileWriterConfig] = FileWriterConfig(),
                 visualizer_config: Optional[VisualizerConfig] = VisualizerConfig()):
        if directory is None:
            directory = 'flow_runs' / pathlib.Path(str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        if directory_suffix is not None:
            directory = directory / directory_suffix
        if directory.exists() and delete_existing:
            print(f'Deleting {str(directory.absolute())}')
            shutil.rmtree(str(directory))

        self.directory = directory.absolute()  # Absolute path helps make logs clear
        self.animation_directory = self.directory / 'animations'
        self.animation_directory.mkdir(exist_ok=True, parents=True)
        self.file_writer = FlowFileWriter(self.directory / 'raw_flow_data', config=file_writer_config)
        self.visualizer = FlowVisualizer(self.directory / 'figures', config=visualizer_config)

        self._step = 0

    def step(self):
        self._step += 1

    def add_scalar(self, tag: str, scalar: float):
        if self.file_writer.config.write_scalars:
            self.file_writer.write_scalar(tag=tag, value=scalar, step=self._step)
        if self.visualizer.config.plot_scalars:
            self.visualizer.add_scalar(tag=tag, value=scalar, step=self._step)

    def add_training_paths(self, paths: torch.Tensor):
        if self.file_writer.config.write_training_paths:
            self.file_writer.write_training_paths(paths=paths, step=self._step)
        if self.visualizer.config.plot_training_paths:
            self.visualizer.plot_forward_paths(paths=paths, global_step=self._step, tag=Tag.train)

    def add_validation_paths(self, paths: torch.Tensor):
        if self.file_writer.config.write_validation_paths:
            self.file_writer.write_validation_paths(paths=paths, step=self._step)
        if self.visualizer.config.plot_validation_paths:
            self.visualizer.plot_forward_paths(paths=paths, global_step=self._step, tag=Tag.validation)

    def add_generative_paths(self, paths: torch.Tensor):
        if self.file_writer.config.write_generative_paths:
            self.file_writer.write_generative_paths(paths=paths, step=self._step)
        if self.visualizer.config.plot_generative_paths:
            self.visualizer.plot_inverse_paths(paths=paths, global_step=self._step, tag=Tag.generative)

    def add_training_reconstructions(self, original_data: torch.Tensor, reconstructed_data: torch.Tensor):
        if self.file_writer.config.write_training_reconstructions:
            self.file_writer.write_training_reconstructions(original_data, reconstructed_data, step=self._step)
        if self.visualizer.config.plot_training_reconstructions:
            self.visualizer.plot_reconstructions(
                original_data,
                reconstructed_data,
                global_step=self._step,
                tag=Tag.train
            )

    def add_validation_reconstructions(self, original_data: torch.Tensor, reconstructed_data: torch.Tensor):
        if self.file_writer.config.write_validation_reconstructions:
            self.file_writer.write_validation_reconstructions(original_data, reconstructed_data, step=self._step)
        if self.visualizer.config.plot_validation_reconstructions:
            self.visualizer.plot_reconstructions(
                original_data,
                reconstructed_data,
                global_step=self._step,
                tag=Tag.validation
            )

    def add_all(self,
                training_paths: torch.Tensor = None,
                validation_paths: torch.Tensor = None,
                training_reconstructions: torch.Tensor = None,
                validation_reconstructions: torch.Tensor = None,
                generative_paths: torch.Tensor = None,
                scalars: dict = None):
        if scalars is None:
            scalars = dict()

        for key, val in scalars:
            self.add_scalar(key, val)

        if training_paths is not None:
            self.add_training_paths(training_paths)
            if training_reconstructions is not None:
                self.add_training_reconstructions(training_paths[0], training_reconstructions)
        if validation_paths is not None:
            self.add_validation_paths(validation_paths)
            if validation_reconstructions is not None:
                self.add_validation_reconstructions(validation_paths[0], validation_reconstructions)
        if generative_paths is not None:
            self.add_generative_paths(generative_paths)

    def animate(self):
        plt.rcParams['animation.ffmpeg_path'] = self.visualizer.config.ffmpeg_path

        if self.visualizer.config.animate_training_paths:
            output_directory = self.animation_directory / 'paths' / Tag.train
            output_directory.mkdir(exist_ok=True, parents=True)
            self.visualizer.animate_forward_paths(
                self.file_writer.paths_train_directory,
                output_directory
            )
        if self.visualizer.config.animate_validation_paths:
            output_directory = self.animation_directory / 'paths' / Tag.validation
            output_directory.mkdir(exist_ok=True, parents=True)
            self.visualizer.animate_forward_paths(
                self.file_writer.paths_validation_directory,
                output_directory
            )
        if self.visualizer.config.animate_generative_paths:
            output_directory = self.animation_directory / 'paths' / Tag.generative
            output_directory.mkdir(exist_ok=True, parents=True)
            self.visualizer.animate_inverse_paths(
                self.file_writer.paths_generative_directory,
                output_directory
            )
        if self.visualizer.config.animate_training_reconstructions:
            output_directory = self.animation_directory / 'reconstructions' / Tag.train
            output_directory.mkdir(exist_ok=True, parents=True)
            self.visualizer.animate_reconstructions(
                self.file_writer.reconstructions_train_directory,
                output_directory
            )
        if self.visualizer.config.animate_validation_reconstructions:
            output_directory = self.animation_directory / 'reconstructions' / Tag.validation
            output_directory.mkdir(exist_ok=True, parents=True)
            self.visualizer.animate_reconstructions(
                self.file_writer.reconstructions_validation_directory,
                output_directory
            )
