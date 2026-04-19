"""Quantum Belief Propagation (BP) preprocessor for variational decoding.

Generates: Section VI.A (Belief Propagation Preprocessing), 
           Figure 8 (bp_convergence.png), Figure 9 (tanner_graph.png)

Novel Contribution: A hybrid BP-Variational architecture where classical 
message-passing provides a high-fidelity soft-prior for the quantum circuit.

This module implements a damped sum-product algorithm on the Tanner graph 
of a stabilizer code to compute marginal error probabilities for each qubit.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

class BeliefPropagator:
    """Belief Propagation (BP) solver for stabilizer codes.

    Implements the sum-product algorithm with damping to handle short cycles 
    in the Tanner graph (factor graph) of the QEC code.

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        The H matrix of the code, shape (n_stabilizers, n_data_qubits).
    max_iterations : int
        Maximum number of message-passing rounds.
    damping : float
        Damping factor in [0, 1] to improve convergence. 
        message = damping * old + (1-damping) * new.
    """

    def __init__(
        self, 
        parity_check_matrix: np.ndarray, 
        max_iterations: int = 50, 
        damping: float = 0.5
    ) -> None:
        self.H = parity_check_matrix.astype(np.uint8)
        self.m, self.n = self.H.shape
        self.max_iterations = max_iterations
        self.damping = damping
        
        # Identify edges in the Tanner graph
        self.edges = []
        for i in range(self.m):
            for j in range(self.n):
                if self.H[i, j]:
                    self.edges.append((i, j))
        
        # Initialize messages: 
        # q_ij: variable j -> check i
        # r_ij: check i -> variable j
        self.q = {edge: 0.0 for edge in self.edges}
        self.r = {edge: 0.0 for edge in self.edges}
        
        logger.info(
            "BeliefPropagator initialised: %d stabilizers, %d qubits, %d edges",
            self.m, self.n, len(self.edges)
        )

    def _get_check_neighbors(self, i: int) -> List[int]:
        return [j for j in range(self.n) if self.H[i, j]]

    def _get_var_neighbors(self, j: int) -> List[int]:
        return [i for i in range(self.m) if self.H[i, j]]

    def compute_log_likelihood_ratios(self, syndrome: np.ndarray, p: float) -> np.ndarray:
        """Compute initial LLRs based on a uniform noise model."""
        llr_0 = np.log((1 - p) / (p / 3.0)) if p > 0 else 10.0
        # For simplicity in this preprocessor, we treat p as the prob of any Pauli error
        # A more complex version would use different LLRs for X, Y, Z
        return np.full(self.n, llr_0)

    def run_bp(
        self, 
        syndrome: np.ndarray, 
        physical_error_rate: float = 0.01
    ) -> Tuple[np.ndarray, bool, int]:
        """Run message passing to estimate qubit error probabilities.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector of length m.
        physical_error_rate : float
            Assumed channel error rate for LLR initialization.

        Returns
        -------
        soft_probs : np.ndarray
            Marginal probability of error for each qubit, shape (n,).
        converged : bool
            Whether the hard decisions satisfy the syndrome.
        n_iterations : int
            Number of iterations performed.
        """
        # Initial LLRs (prior beliefs)
        # L_j = log( P(e_j=0) / P(e_j=1) )
        prior_llr = self.compute_log_likelihood_ratios(syndrome, physical_error_rate)
        
        # Initialize variable-to-check messages with priors
        for i, j in self.edges:
            self.q[(i, j)] = prior_llr[j]
            self.r[(i, j)] = 0.0

        converged = False
        final_iter = 0
        
        for iteration in range(self.max_iterations):
            final_iter = iteration + 1
            
            # 1. Check-to-variable messages (r_ij)
            # r_ij = 2 * atanh( prod_{j' in N(i)\j} tanh(q_ij' / 2) ) * (-1)^s_i
            new_r = {}
            for i, j in self.edges:
                neighbors = self._get_check_neighbors(i)
                prod = 1.0
                for j_prime in neighbors:
                    if j_prime == j:
                        continue
                    prod *= np.tanh(self.q[(i, j_prime)] / 2.0)
                
                # Clip product for numerical stability
                prod = np.clip(prod, -0.999, 0.999)
                val = 2.0 * np.arctanh(prod)
                if syndrome[i] == 1:
                    val *= -1.0
                
                # Apply damping
                new_r[(i, j)] = self.damping * self.r[(i, j)] + (1 - self.damping) * val
            self.r = new_r

            # 2. Variable-to-check messages (q_ij)
            # q_ij = L_j + sum_{i' in N(j)\i} r_i'j
            new_q = {}
            for i, j in self.edges:
                neighbors = self._get_var_neighbors(j)
                val = prior_llr[j] + sum(self.r[(i_prime, j)] for i_prime in neighbors if i_prime != i)
                new_q[(i, j)] = self.damping * self.q[(i, j)] + (1 - self.damping) * val
            self.q = new_q

            # 3. Hard decision and convergence check
            posterior_llr = np.zeros(self.n)
            for j in range(self.n):
                neighbors = self._get_var_neighbors(j)
                posterior_llr[j] = prior_llr[j] + sum(self.r[(i, j)] for i in neighbors)
            
            hard_decisions = (posterior_llr < 0).astype(np.uint8)
            current_syndrome = (self.H @ hard_decisions) % 2
            if np.array_equal(current_syndrome, syndrome):
                converged = True
                break

        # Compute soft probabilities from posterior LLRs
        # P(e=1) = 1 / (1 + exp(LLR))
        soft_probs = 1.0 / (1.0 + np.exp(posterior_llr))
        
        return soft_probs, converged, final_iter

    def compute_soft_probabilities(self, syndrome: np.ndarray, p: float = 0.01) -> np.ndarray:
        """Helper to get just the soft probabilities."""
        probs, _, _ = self.run_bp(syndrome, p)
        return probs

    def plot_tanner_graph(self, save_path: str = "results/figures/tanner_graph.png") -> None:
        """Visualize the factor graph of the code."""
        G = nx.Graph()
        
        # Add stabilizer nodes (check nodes)
        stabilizer_nodes = [f"S{i}" for i in range(self.m)]
        G.add_nodes_from(stabilizer_nodes, bipartite=0)
        
        # Add qubit nodes (variable nodes)
        qubit_nodes = [f"Q{j}" for j in range(self.n)]
        G.add_nodes_from(qubit_nodes, bipartite=1)
        
        # Add edges
        for i, j in self.edges:
            G.add_edge(f"S{i}", f"Q{j}")
            
        pos = nx.bipartite_layout(G, stabilizer_nodes)
        
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=stabilizer_nodes, node_color='r', node_shape='s', label='Stabilizers')
        nx.draw_networkx_nodes(G, pos, nodelist=qubit_nodes, node_color='b', node_shape='o', label='Qubits')
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Tanner Graph (Stabilizer Factor Graph)")
        plt.legend()
        plt.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved Tanner graph plot to %s", save_path)

    def convergence_analysis(
        self, 
        n_syndromes: int = 100, 
        p: float = 0.05,
        save_path: str = "results/figures/bp_convergence.png"
    ) -> None:
        """Plot convergence rate vs iteration count."""
        # This would normally sample from a noise model, 
        # for this module we assume a simple error generator
        convergence_counts = np.zeros(self.max_iterations + 1)
        
        for _ in range(n_syndromes):
            # Generate random error
            err = (np.random.random(self.n) < p).astype(np.uint8)
            syn = (self.H @ err) % 2
            
            _, converged, iters = self.run_bp(syn, p)
            if converged:
                convergence_counts[iters:] += 1
        
        rates = convergence_counts / n_syndromes
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(rates)), rates, marker='o', color='#0173b2')
        plt.xlabel("BP Iterations")
        plt.ylabel("Convergence Rate")
        plt.title(f"BP Convergence Analysis (p={p})")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved BP convergence plot to %s", save_path)
