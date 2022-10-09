import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rcParams['figure.facecolor'] = 'dimgrey'
from scipy.spatial import distance
from itertools import combinations
import time 


class AntColonyOptimizer:
    def __init__(self, ants = 100, evaporation_rate = 0.15, intensification = 0.15, alpha = 1.0, beta = 0.0, beta_evaporation_rate = 0.0,
                 choose_best = 0.1):
            
        """
        Ant colony optimizer. Finds the path that either maximizes or minimizes distance traveled between nodes. 
        This optimizer is devoted to solving Traveling Salesman Problem (TSP).

        :param ants: Number of ants to traverse the graph.
        :param evaporation_rate: Rate at which pheromone evaporates.
        :param intensification: Constant value added to the best path.
        :param alpha: Weight of pheromone.
        :param beta: Weight of heuristic (1/distance).
        :param beta_evaporation_rate: Rate at which beta decays.
        :param choose_best: Probability to choose the best path.

        Created by Radosław Sergiusz Jasiewicz. Enjoy :)
        """

        # Parameters
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_intensification = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        self.choose_best = choose_best

        # Matrices
        self.distance_matrix = None
        self.pheromone_matrix = None
        self.heuristic_matrix = None
        self.probability_matrix = None

        self.spatial_map = None
        self.num_nodes = None
        self.list_of_nodes = None

        # Internal statistics
        self.global_best = None
        self.best_series = None
        self.best_path = None
        self.best = None

        # Optimizer's status
        self.fitted = False
        self.fit_time = None
        self.mode = None

        # Early stopping 
        self.converged = False
        self.stopped_at_iteration = None

    def __str__(self):

        string = "Ant Colony Optimizer"
        string += "\n------------------------"
        string += "\nDesigned to solve travelling salesman problem. Optimizes either the minimum or maximum distance travelled."
        string += "\n------------------------"
        string += f"\nNumber of ants:\t\t\t\t{self.ants}"
        string += f"\nEvaporation rate:\t\t\t{self.evaporation_rate}"
        string += f"\nIntensification factor:\t\t\t{self.pheromone_intensification}"
        string += f"\nAlpha Heuristic:\t\t\t{self.heuristic_alpha}"
        string += f"\nBeta Heuristic:\t\t\t\t{self.heuristic_beta}"
        string += f"\nBeta Evaporation Rate:\t\t\t{self.beta_evaporation_rate}"
        string += f"\nChoose Best Procentage:\t\t\t{self.choose_best}"
        string += "\n------------------------"

        if self.fitted:
            string += "\n\nThis optimizer has been fitted."
        else:
            string += "\n\nThis optimizer has NOT been fitted."
        return string
    
    def __initialize(self, spatial_map: list):
        """
        Initializes various matrices and checks given list of points.
        """
        for _ in spatial_map:
            assert len(_) == 2, "These are not valid points! Maybe check them? :)"

        # Create distance matrix
        self.spatial_map = spatial_map
        self.num_nodes = len(self.spatial_map)
        self.distance_matrix = distance.cdist(spatial_map, spatial_map)

        # Create pheromone matrix 
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes)) 

        # Create heuristic matrix 
        self.heuristic_matrix = np.divide(1, self.distance_matrix, out=np.zeros(self.distance_matrix.shape, dtype=float), where= self.distance_matrix != 0)

        # Remove diagonals from matrices 
        np.fill_diagonal(self.pheromone_matrix, val=0)
        np.fill_diagonal(self.heuristic_matrix, val=0)

        # Create probability matrix 
        self.probability_matrix = (
            (self.pheromone_matrix ** self.heuristic_alpha) 
            * 
            (self.heuristic_matrix ** self.heuristic_beta)
        )

        # Fill the list of nodes
        self.list_of_nodes = np.arange(0, self.num_nodes, dtype=int)

    def __reset_list_of_nodes(self):
        """
        Reset the list of all nodes for the next iteration.
        """
        self.list_of_nodes = np.arange(0, self.num_nodes, dtype=int)

    def __update_probabilities(self):
        """
        Each iteration probability matrix needs to be updated. This function does exactly that.
        """
        self.probability_matrix = (
            (self.pheromone_matrix ** self.heuristic_alpha) 
            * 
            (self.heuristic_matrix ** self.heuristic_beta)
        )

    def __travel_to_next_node_from(self, current_node: int):
        """
        Chooses the next node based on probabilities. 

        :param current_node: The node an ant is currently at.
        """
        numerator = self.probability_matrix[current_node, self.list_of_nodes]

        if np.random.random() < self.choose_best:
            next_node = np.argmax(numerator)
        else:
            denominator = np.sum(numerator)
            probabilities = numerator / denominator
            next_node = np.random.choice(range(len(probabilities)), p=probabilities)
        return next_node

    def __remove_node(self, node: int):
        """
        Removes the node after an ant has traveled through it.
        """
        index = np.where(self.list_of_nodes == node)[0][0]

        self.list_of_nodes[index] = -1
        self.list_of_nodes = self.list_of_nodes[self.list_of_nodes != -1]

    def __evaluate(self, iteration: int, paths: list, mode: str):
        """
        Evaluates ants' solution from iteration. Given all paths that ants have chosen, pick the best one.

        :param iteration: Iteration's number.
        :param paths: Ants' paths form iteration.
        :param mode: Chosen mode. 
        :return: Best path and best score.
        """
        paths = paths[iteration, :, :]
        scores = np.zeros(len(paths))
        
        for i, path in enumerate(paths):
            score = 0
            for index in range(len(path) - 1):
                score += self.distance_matrix[path[index], path[index+1]]
            scores[i] = score

        if mode == 'min':
            best = np.argmin(scores)
        elif mode == 'max':
            best = np.argmax(scores)

        return paths[best], scores[best]

    def __evaporate(self):
        """
        Evaporates pheromone every iteration. Also evaporates beta parameter (optional).
        """
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        self.heuristic_beta *= (1 - self.beta_evaporation_rate)

    def __intensify(self, best_path: list):
        """
        Adds constant value (enlarges pheromone value) to the best path.
        """
        for i in range(len(best_path) - 1):
            self.pheromone_matrix[best_path[i], best_path[i+1]] += self.pheromone_intensification

    def plot(self, show_global = True, figsize = (10,5), dpi = 200):
        """
        Plots performance over iterations. If ACO has NOT been fitted returns None.
        """
        if not self.fitted:
            print("Ant Colony Optimizer not fitted! There is nothing to plot.")
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if show_global:
                ax.plot(
                    self.global_best[:self.stopped_at_iteration],
                    lw = 0.5,
                    color = "lime",
                    alpha=0.5,
                    zorder = 0,
                    label = r"    Current best path"
                )

            ax.plot(
                self.best_series[:self.stopped_at_iteration],
                linewidth = 0.5,
                c = "blue",
                alpha=0.5,
                label = r"Best path from iteration",
                zorder = 1
                )
            
            ax.scatter(
                x = np.arange(0, self.stopped_at_iteration, 1),
                y = self.best_series[:self.stopped_at_iteration],
                marker="o",
                c="blue",
                s=2,
                zorder=1
            )

            ax.set_xlabel(r"$\bf{Iteration}$ $\bf{number}$") ; ax.set_ylabel(r"$\bf{Performance}$")
            ax.set_xlim(0, self.stopped_at_iteration - 1)
            ax.set_xticks(np.arange(0, self.stopped_at_iteration + 1, 10))

            ax.set_facecolor('lightgrey')

            if self.mode == 'min':
                ax.legend(loc="upper right", frameon=False)
            else:
                ax.legend(loc="lower right", frameon=False)
            plt.show()

    def fit(self, spatial_map: list, iterations = 100, mode = 'min', conv_crit = 20, verbose = True, decimal = 2):
        """
        The core function of the optimizer. It fits ACO to a specific map.

        :param spatial_map: List of positions (x, y) of the nodes.
        :param iterations: Number of iterations.
        :param mode: Find min or max distance between nodes.
        :param conv_crit: Convergence criterion. The function stops after that many repeated scores. 
        :param verbose: If enabled ACO informs you about the progress.
        :param decimal: Number of decimal places. 
        """
        start = time.time()
        self.__initialize(np.array(spatial_map)) ; self.mode = mode
        num_equal = 0

        if verbose: print(f"{self.num_nodes} nodes were given. Beginning ACO Optimization with {iterations} iterations...\n")

        self.best_series = np.zeros(iterations, dtype=float)
        self.global_best = np.zeros(iterations, dtype=float)

        paths = np.empty((iterations, self.ants, self.num_nodes + 1), dtype=int)
        path = np.empty(self.num_nodes + 1, dtype=int)

        for iteration in range(iterations):
            start_iter = time.time()

            for ant in range(self.ants):
                current_node = self.list_of_nodes[np.random.randint(0, self.num_nodes)]
                start_node = current_node
                node = 0
                while True:
                    path[node] = current_node
                    self.__remove_node(current_node)
                    if len(self.list_of_nodes) != 0:
                        next_node = self.__travel_to_next_node_from(current_node)
                        current_node = self.list_of_nodes[next_node]
                        node += 1
                    else:
                        break 

                path[node + 1] = start_node # go back to start 
                self.__reset_list_of_nodes() 
                paths[iteration, ant] = path # add path to batch 
                path = np.empty(self.num_nodes + 1, dtype=int) # reset path 

            best_path, best_score = self.__evaluate(iteration, paths, mode)

            if iteration == 0:
                best_score_so_far = best_score
                self.best_path = best_path
            else:
                if mode == 'min':
                    if best_score < best_score_so_far:
                        best_score_so_far = best_score
                        self.best_path = best_path
                elif mode == 'max':
                    if best_score > best_score_so_far:
                        best_score_so_far = best_score
                        self.best_path = best_path

            if best_score == best_score_so_far:
                num_equal += 1
            else:
                num_equal = 0

            self.best_series[iteration] = best_score
            self.global_best[iteration] = best_score_so_far
            self.__evaporate()
            self.__intensify(best_path)
            self.__update_probabilities()

            if verbose: print(
                f"Iteration {iteration}/{iterations} | Score: {round(best_score, decimal)} | Best so far: {round(best_score_so_far, decimal)} | {round(time.time() - start_iter, decimal)} s"
            )

            if (best_score == best_score_so_far) and (num_equal == conv_crit):
                self.converged = True ; self.stopped_at_iteration = iteration
                if verbose: print("\nConvergence criterion has been met. Stopping....")
                break

        if not self.converged: self.stopped_at_iteration = iterations
        self.fit_time = round(time.time() - start)
        self.fitted = True

        if mode == 'min':

            if self.converged:
                self.best_series = self.best_series[self.best_series > 0]
            else:
                pass

            self.best = self.best_series[np.argmin(self.best_series)]

        if mode == 'max':

            if self.converged:
                self.best_series = self.best_series[self.best_series > 0]
            else:
                pass

            self.best = self.best_series[np.argmax(self.best_series)]

        if verbose: print(
            f"\nACO fitted. Runtime: {self.fit_time // 60} minute(s). Best score: {round(self.best, decimal)} | Mode: {mode}"
        )
        
    def show_graph(self, fitted = True, figsize = (10,5), dpi = 200):
        """
        Shows the graph of nodes that ACO is working on.

        :param fitted: If enabled it shows the best path that the optimizer has found.
        """
        fig, ax = plt.subplots(figsize=figsize, dpi = dpi)

        for node, other_node in combinations(self.spatial_map, r=2):
            ax.plot(
                [node[0], other_node[0]],
                [node[1], other_node[1]],
                color="blue",
                linewidth=0.2,
                alpha=0.1,
                zorder=0
                )

        for index in range(self.num_nodes):
            ax.scatter(
                x=self.spatial_map[index,0],
                y=self.spatial_map[index,1],
                linewidth=0.5,
                marker="o",
                s=8,
                edgecolor="lime",
                c="black",
                zorder=2
            )

        if self.fitted and fitted:
            for i in range(len(self.best_path) - 1):
                ax.plot(
                    [self.spatial_map[self.best_path[i]][0], self.spatial_map[self.best_path[i+1]][0]],
                    [self.spatial_map[self.best_path[i]][1], self.spatial_map[self.best_path[i+1]][1]],
                    color = "lime",
                    linewidth = 0.5,
                    alpha = 1,
                    zorder = 1
                )

        ax.set_title(r"$\bf{Graph}$ $\bf{of}$ $\bf{nodes}$")
        ax.axis('off')
        plt.show()

    def get_result(self):
        """
        :return: Tuple consisted of the best path, best distance, fit time, and list of each generation's best distance.
        """
        return self.best_path, self.best, self.fit_time, self.best_series










