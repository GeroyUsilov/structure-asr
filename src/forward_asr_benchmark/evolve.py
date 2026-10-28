"""Forward evolution down a phylogeny with complete node-level ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import numpy as np

from .dca import DcaModel
from .scoring import ScoreResult, SequenceScorer


@dataclass(frozen=True)
class Proposal:
    sequence: str
    site: int
    old_state: str
    new_state: str
    log_q_forward: float
    log_q_reverse: float


class ProposalKernel(Protocol):
    def propose(self, sequence: str, rng: np.random.Generator) -> Proposal: ...


@dataclass
class UniformSubstitutionProposal:
    alphabet: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

    def propose(self, sequence: str, rng: np.random.Generator) -> Proposal:
        site = int(rng.integers(len(sequence)))
        old = sequence[site]
        candidates = [token for token in self.alphabet if token != old]
        if not candidates:
            raise ValueError("proposal alphabet has no alternative state")
        new = str(rng.choice(candidates))
        candidate = sequence[:site] + new + sequence[site + 1 :]
        log_q = -np.log(len(sequence)) - np.log(len(candidates))
        return Proposal(candidate, site, old, new, float(log_q), float(log_q))


@dataclass
class DcaConditionalProposal:
    model: DcaModel

    def propose(self, sequence: str, rng: np.random.Generator) -> Proposal:
        site = int(rng.integers(len(sequence)))
        old = sequence[site]
        new, conditional_log_q = self.model.sample_conditional(sequence, site, rng)
        candidate = sequence[:site] + new + sequence[site + 1 :]
        log_probabilities = self.model.conditional_log_probabilities(sequence, site)
        reverse_log_q = float(log_probabilities[self.model.index[old]])
        site_log_q = -np.log(len(sequence))
        return Proposal(
            candidate,
            site,
            old,
            new,
            float(site_log_q + conditional_log_q),
            float(site_log_q + reverse_log_q),
        )


@dataclass
class ProfileConditionalProposal:
    """Pseudo-Gibbs proposal from an OFS/exact masked residue profile.

    A site is selected uniformly, matching the DCA evolver, and its state is
    sampled from the profile conditional at that site. The profile columns must
    follow ``alphabet``. ``proposal_temperature`` can flatten (>1) or sharpen
    (<1) the amino-acid distribution.

    Unlike a Potts model, masked-LM conditionals need not correspond to a
    consistent joint distribution. Use kernel="ofs_pseudogibbs" to reproduce
    the DCA-like always-accept procedure, or kernel="metropolis" to target an
    explicit composite energy with the calculated Hastings correction.
    """

    profile_predictor: Callable[[str], np.ndarray]
    alphabet: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
    proposal_temperature: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be positive")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("proposal alphabet must contain unique tokens")

    def _probabilities(self, sequence: str) -> np.ndarray:
        profile = np.asarray(self.profile_predictor(sequence), dtype=float)
        expected = (len(sequence), len(self.alphabet))
        if profile.shape != expected:
            raise ValueError(f"profile shape {profile.shape} does not match {expected}")
        if np.any(profile < 0) or np.any(profile.sum(axis=1) <= 0):
            raise ValueError("profile must contain nonnegative, nonempty rows")
        probabilities = np.maximum(profile, self.epsilon)
        probabilities = probabilities ** (1.0 / self.proposal_temperature)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def propose(self, sequence: str, rng: np.random.Generator) -> Proposal:
        site = int(rng.integers(len(sequence)))
        old = sequence[site]
        if old not in self.alphabet:
            raise ValueError(f"sequence token {old!r} is not in profile alphabet")
        probabilities = self._probabilities(sequence)
        new_index = int(rng.choice(len(self.alphabet), p=probabilities[site]))
        new = self.alphabet[new_index]
        candidate = sequence[:site] + new + sequence[site + 1 :]
        reverse_probabilities = self._probabilities(candidate)
        old_index = self.alphabet.index(old)
        site_log_q = -np.log(len(sequence))
        return Proposal(
            sequence=candidate,
            site=site,
            old_state=old,
            new_state=new,
            log_q_forward=float(site_log_q + np.log(probabilities[site, new_index])),
            log_q_reverse=float(site_log_q + np.log(reverse_probabilities[site, old_index])),
        )


@dataclass
class MaskedSiteConditionalProposal:
    """Exact masked-LM proposal requiring one model pass per update.

    A site is selected uniformly and ``site_profile_predictor`` returns the
    20-amino-acid conditional distribution obtained by masking only that site.
    Because the forward and reverse moves mask the same residue position, they
    share the same context and therefore the same conditional profile; no
    second model call is needed to calculate the reverse proposal probability.
    """

    site_profile_predictor: Callable[[str, int], np.ndarray]
    alphabet: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
    proposal_temperature: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.proposal_temperature <= 0:
            raise ValueError("proposal_temperature must be positive")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("proposal alphabet must contain unique tokens")

    def propose(self, sequence: str, rng: np.random.Generator) -> Proposal:
        site = self.choose_site(sequence, rng)
        return self.complete(sequence, site, self.site_profile_predictor(sequence, site), rng)

    def choose_site(self, sequence: str, rng: np.random.Generator) -> int:
        """First RNG draw of an update; split out so callers can batch model passes."""
        site = int(rng.integers(len(sequence)))
        if sequence[site] not in self.alphabet:
            raise ValueError(f"sequence token {sequence[site]!r} is not in masked-LM alphabet")
        return site

    def complete(
        self, sequence: str, site: int, profile, rng: np.random.Generator
    ) -> Proposal:
        """Temper the masked-site profile and draw the new state (second RNG draw)."""
        old = sequence[site]
        probabilities = np.asarray(profile, dtype=float)
        expected = (len(self.alphabet),)
        if probabilities.shape != expected:
            raise ValueError(
                f"site profile shape {probabilities.shape} does not match {expected}"
            )
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("site profile must contain nonnegative probability mass")
        probabilities = np.maximum(probabilities, self.epsilon)
        probabilities = probabilities ** (1.0 / self.proposal_temperature)
        probabilities = probabilities / probabilities.sum()
        new_index = int(rng.choice(len(self.alphabet), p=probabilities))
        new = self.alphabet[new_index]
        old_index = self.alphabet.index(old)
        candidate = sequence[:site] + new + sequence[site + 1 :]
        site_log_q = -np.log(len(sequence))
        return Proposal(
            sequence=candidate,
            site=site,
            old_state=old,
            new_state=new,
            log_q_forward=float(site_log_q + np.log(probabilities[new_index])),
            log_q_reverse=float(site_log_q + np.log(probabilities[old_index])),
        )


@dataclass(frozen=True)
class SimulationSettings:
    mutation_scale: float = 1.0
    temperature: float = 1.0
    seed: int = 1
    kernel: str = "metropolis"

    def __post_init__(self) -> None:
        if self.mutation_scale < 0:
            raise ValueError("mutation_scale must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.kernel not in {
            "metropolis",
            "dca_gibbs",
            "ofs_pseudogibbs",
            "exact_mlm_gibbs",
        }:
            raise ValueError(
                "kernel must be 'metropolis', 'dca_gibbs', 'ofs_pseudogibbs', "
                "or 'exact_mlm_gibbs'"
            )


@dataclass(frozen=True)
class MutationEvent:
    parent: str
    child: str
    attempt: int
    site: int
    old_state: str
    new_state: str
    accepted: bool
    log_acceptance: float
    energy_before: float
    energy_after: float


@dataclass
class EvolutionResult:
    sequences: dict[str, str] = field(default_factory=dict)
    scores: dict[str, ScoreResult] = field(default_factory=dict)
    events: list[MutationEvent] = field(default_factory=list)


class ForwardSimulator:
    def __init__(
        self,
        scorer: SequenceScorer,
        proposal: ProposalKernel,
        settings: SimulationSettings,
    ) -> None:
        self.scorer = scorer
        self.proposal = proposal
        self.settings = settings
        if settings.kernel == "dca_gibbs" and not isinstance(proposal, DcaConditionalProposal):
            raise ValueError("dca_gibbs requires DcaConditionalProposal")
        if settings.kernel == "ofs_pseudogibbs" and not isinstance(proposal, ProfileConditionalProposal):
            raise ValueError("ofs_pseudogibbs requires ProfileConditionalProposal")
        if settings.kernel == "exact_mlm_gibbs" and not isinstance(
            proposal, MaskedSiteConditionalProposal
        ):
            raise ValueError("exact_mlm_gibbs requires MaskedSiteConditionalProposal")

    @staticmethod
    def _stochastic_round(value: float, rng: np.random.Generator) -> int:
        base = int(np.floor(value))
        return base + int(rng.random() < value - base)

    def simulate(
        self,
        tree,
        root_sequence: str,
        *,
        initial_result: EvolutionResult | None = None,
        start_edge: int = 0,
        edge_completed: Callable[[int, int, EvolutionResult], None] | None = None,
    ) -> EvolutionResult:
        root_name = str(tree.root.name)
        if initial_result is None:
            root_score = self.scorer.score(root_sequence)
            result = EvolutionResult(
                sequences={root_name: root_sequence},
                scores={root_name: root_score},
            )
        else:
            result = initial_result
            if result.sequences.get(root_name) != root_sequence:
                raise ValueError("checkpoint root sequence does not match requested root")
        seed_sequence = np.random.SeedSequence(self.settings.seed)
        edges = [
            (parent, child)
            for parent in tree.find_clades(order="preorder")
            for child in parent.clades
        ]
        if not 0 <= start_edge <= len(edges):
            raise ValueError(f"start_edge {start_edge} is outside 0..{len(edges)}")
        edge_seeds = seed_sequence.spawn(len(edges))

        for edge_index, ((parent, child), edge_seed) in enumerate(zip(edges, edge_seeds)):
            if edge_index < start_edge:
                continue
            parent_name = str(parent.name)
            child_name = str(child.name)
            if parent_name not in result.sequences:
                raise RuntimeError("tree traversal reached a child before its parent")
            rng = np.random.default_rng(edge_seed)
            sequence = result.sequences[parent_name]
            score = result.scores[parent_name]
            attempts = self._stochastic_round(
                self.settings.mutation_scale * len(sequence) * float(child.branch_length or 0.0),
                rng,
            )
            for attempt in range(1, attempts + 1):
                move = self.proposal.propose(sequence, rng)
                proposed_score = self.scorer.score(move.sequence)
                if self.settings.kernel in {
                    "dca_gibbs",
                    "ofs_pseudogibbs",
                    "exact_mlm_gibbs",
                }:
                    log_acceptance = 0.0
                    accepted = True
                else:
                    log_acceptance = min(
                        0.0,
                        -(proposed_score.energy - score.energy) / self.settings.temperature
                        + move.log_q_reverse
                        - move.log_q_forward,
                    )
                    accepted = bool(np.log(rng.random()) < log_acceptance)
                result.events.append(
                    MutationEvent(
                        parent=parent_name,
                        child=child_name,
                        attempt=attempt,
                        site=move.site,
                        old_state=move.old_state,
                        new_state=move.new_state,
                        accepted=accepted,
                        log_acceptance=float(log_acceptance),
                        energy_before=score.energy,
                        energy_after=proposed_score.energy,
                    )
                )
                if accepted:
                    sequence = move.sequence
                    score = proposed_score
            result.sequences[child_name] = sequence
            result.scores[child_name] = score
            if edge_completed is not None:
                edge_completed(edge_index + 1, len(edges), result)
        return result
