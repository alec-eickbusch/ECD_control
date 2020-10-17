import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42})


class VisualizationMixin:
    def plot_initial_state(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        state = tfq.tf2qt(self.initial_state)
        plot_wigner(
            state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar
        )

    def plot_final_state(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        state = tfq.tf2qt(
            self.final_state(self.betas_rho, self.betas_angle, self.phis, self.thetas)
        )
        plot_wigner(
            state,
            contour=contour,
            fig=fig,
            ax=ax,
            max_alpha=max_alpha,
            cbar=cbar,
        )

    def plot_target_state(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        state = tfq.tf2qt(self.target_state)
        plot_wigner(
            state,
            contour=contour,
            fig=fig,
            ax=ax,
            max_alpha=max_alpha,
            cbar=cbar,
        )

    def plot_state(self, i=0, contour=True, fig=None, ax=None, max_alpha=6, cbar=False):
        state = tfq.tf2qt(self.state(i=i))
        plot_wigner(
            state,
            contour=contour,
            fig=fig,
            ax=ax,
            max_alpha=max_alpha,
            cbar=cbar,
        )
