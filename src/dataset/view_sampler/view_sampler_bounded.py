from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np
import os

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    random: bool = False
    extra: bool = False


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        phase: int = 1,
        test_fvs: bool = False,
        path: str = None,
        n_views = None,
        coverage = None,
        interval = None,
    ):
        num_views, _, _ = extrinsics.shape

        if coverage is not None:
            # print('interval:', interval)
            num_try = 0
            while num_try < 10:
                M = n_views if self.cfg.random else self.num_context_views
                available = np.where(interval > M*2)[0]
                try:
                    cluster_id = np.random.choice(available)
                except:
                    print('++++++++++++++++=Not enough views in scene:', scene)
                    exit(0)
                start = np.sum(interval[:cluster_id])
                length = interval[cluster_id]

                # try:
                sub_coverage = np.array(coverage)[start:start+length, start:start+length]
                # except:
                #     print('coverage:', coverage)
                #     exit(0)

                N = sub_coverage.shape[0]
                if M > N:
                    raise ValueError("M cannot be greater than N")

                # Choose a random start index
                start_index = np.random.randint(1, max(2, N-M*5))
                sampled_indices = [start+start_index]

                # Attempt to find the next best view
                current_index = start_index
                coverages = []
                while len(sampled_indices) < M and current_index < N-2:
                    # Calculate next index with wrap-around
                    next_index = np.abs(sub_coverage[current_index, current_index + 2:current_index + 20] - 0.8).argmin() + current_index + 2

                    coverages.append(sub_coverage[current_index, next_index])
                    sampled_indices.append(start + next_index)
                    current_index = next_index
                
                if len(sampled_indices) == M and all([c > 0.6 for c in coverages]):
                    break
                num_try += 1

            # Check if we have less views than needed
            if len(sampled_indices) < M:
                # Fill up remaining with the best possible option, maintaining intervals
                start_index = 1
                interval = min(N // M, 10)
                sampled_indices = [start+start_index+i * interval for i in range(M)]
                coverages = [sub_coverage[i-start, i-start+interval] for i in sampled_indices[:-1]]

            if M in [2,3]:
                num_target_views = 4
            else:
                num_target_views = M - 1
            rest_indices = [i for i in range(start+start_index, sampled_indices[-1]) if i not in sampled_indices]
            target_indices = np.random.choice(rest_indices, min(num_target_views, len(rest_indices)), replace=False)
            target_indices = np.sort(target_indices)

            return (
                torch.tensor([sampled_indices], dtype=torch.int64, device=device),
                torch.from_numpy(target_indices),
                0
            )

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
           # When testing, always use the full gap.
           max_gap = self.cfg.max_distance_between_context_views
           min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()
        
        if self.cfg.random:
            num_context_views = n_views
        else:
            num_context_views = self.num_context_views
        if (num_context_views > (num_views-1) // context_gap + 1) and not self.cfg.random:
            raise ValueError("Not enough views for the context views!")
        num_context_views = min(num_context_views, (num_views-1) // context_gap + 1)
        index_context_left = torch.randint(
                num_views if self.cameras_are_circular else num_views - context_gap*(num_context_views+phase-2),
                size=tuple(),
                device=device,
            ).item()
        index_start = index_context_left
        
        context_views_all = []
        index_target = []
        if num_context_views == 2:
            per_size = 4
        elif num_context_views == 3:
            per_size = 2
        else:
            per_size = 1
        for p in range(phase):
            # Pick the left and right context indices.
            
            context_views = [index_context_left]
            for i in range(num_context_views-1):
                index_context_right = context_views[i] + context_gap

                if self.is_overfitting:
                    index_context_left *= 0
                    index_context_right *= 0
                    index_context_right += max_gap

                # Pick the target view indices.
                index_target.append(torch.randint(
                        context_views[i] + self.cfg.min_distance_to_context_views,
                        index_context_right - self.cfg.min_distance_to_context_views,
                        size=(per_size,),
                        device=device,
                    ))
                context_views.append(index_context_right)
 
            index_context_left += context_gap
            context_views_all.append(context_views)

            # Apply modulo for circular datasets.
            if self.cameras_are_circular:
                index_target %= num_views
                index_context_right %= num_views
        
        return (
            torch.tensor(context_views_all, dtype=torch.int64, device=device),
            torch.cat(index_target),
            0
        )

    @property
    def num_context_views(self) -> int:
        # return 2
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
