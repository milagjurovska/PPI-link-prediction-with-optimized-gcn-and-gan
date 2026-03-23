from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import GeneticAlgorithm, ParticleSwarmOptimization, ArtificialBeeColonyAlgorithm
from niapy.algorithms.other import SimulatedAnnealing, HillClimbAlgorithm, RandomSearch
from gcn import *
from gan import *
from data_processing import train_data, test_data
from niapy.algorithms.algorithm import Individual
import time


# Search space dimensions:
#   x[0] = hidden_channels  in [64,   512]
#   x[1] = lr               in [1e-5, 0.1]
#   x[2] = dropout          in [0.0,  0.8]
#   x[3] = weight_decay     in [1e-7, 1e-2]
#   x[4] = beta1            in [0.3,  0.9]

class GANHyperparameterProblem(Problem):
    def __init__(self):
        super().__init__(
            dimension=5,
            lower=[64, 1e-5, 0.0, 1e-7, 0.3],
            upper=[512, 0.1, 0.8, 1e-2, 0.9],
            dtype=float
        )
        self.last_f1 = None
        self.last_auc = None
        self.last_loss = None
        self.last_ndcg = None

    def _evaluate(self, x):
        hidden_channels = int(x[0])
        lr = float(x[1])
        dropout = float(x[2])
        weight_decay = float(x[3])
        beta1 = float(x[4])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        generator = Generator(in_channels=5, hidden_channels=hidden_channels).to(device)
        discriminator = Discriminator(hidden_channels=hidden_channels).to(device)

        generator.dropout = torch.nn.Dropout(dropout)

        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=weight_decay)

        total_d_loss = 0
        total_g_loss = 0
        for _ in range(5):
            d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
            if d_loss > 1.0 or g_loss > 1.0:
                return float('inf')
            total_d_loss += d_loss
            total_g_loss += g_loss

        test_probs = GANtest(generator, discriminator, test_data)
        auc, f1, ndcg = evaluate_model(test_probs, test_data.edge_label)

        avg_loss = (total_d_loss + total_g_loss) / 20

        self.last_f1 = f1
        self.last_auc = auc
        self.last_loss = avg_loss
        self.last_ndcg = ndcg

        return -f1


def _extract_gan_result(best_solution, best_score, problem, duration):
    return {
        "best_params": {
            "hidden_channels": int(best_solution[0]),
            "lr": best_solution[1],
            "dropout": best_solution[2],
            "weight_decay": best_solution[3],
            "beta1": best_solution[4]
        },
        "f1": -best_score,
        "auc": problem.last_auc,
        "loss": problem.last_loss,
        "ndcg": problem.last_ndcg,
        "time_taken": duration
    }


def run_gan_ga():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = GeneticAlgorithm(
        population_size=10,
        crossover_rate=0.8,
        mutation_rate=0.2,
        individual_type=Individual
    )

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)


def run_gan_pso():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = ParticleSwarmOptimization(
        population_size=10,
        c1=2.0,
        c2=2.0,
        w=0.7
    )

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)


def run_gan_abc():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = ArtificialBeeColonyAlgorithm(
        population_size=10,
        limit=100
    )

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)


def run_gan_sa():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = SimulatedAnnealing(
        t_min=0.001,
        t_max=1000.0,
        alpha=0.99
    )

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)


def run_gan_hc():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = HillClimbAlgorithm(
        delta=0.1
    )

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)


def run_gan_ra():
    problem = GANHyperparameterProblem()
    task = Task(problem=problem, max_evals=30)

    algo = RandomSearch()

    start_time = time.time()
    best_solution, best_score = algo.run(task)
    duration = time.time() - start_time
    return _extract_gan_result(best_solution, best_score, problem, duration)
