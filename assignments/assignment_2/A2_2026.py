"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec import (
    Archive,
    EA,
    EAOperation,
    FloatMutator,
    Individual,
    Population,
)

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options
CROSSOVER = "n_point" # "uniform" | "n_point" 
N_CUTS = 4
STARTER_SIZE = 20
TARGET_POPULATION = 20
STEPS = 100


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = [np.asarray(w) for w in weights]

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    inputs = data.qpos

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)).tolist(),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)).tolist(),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #
def _flatten_genotype(
    genotype: list,
) -> tuple[npt.NDArray[np.float32], list[tuple[int, ...]]]:
    """Flattens a list of weight matrices (or lists) into a single 1D array."""
    shapes = [np.asarray(arr).shape for arr in genotype]
    flat_array = np.concatenate(
        [np.asarray(arr, dtype=np.float32).ravel() for arr in genotype]
    )
    return flat_array, shapes


def _unflatten_genotype(
    flat_array: npt.NDArray[np.float32] | list[float], 
    shapes: list[tuple[int, ...]]
) -> list[list[float]]:
    """Reshapes a 1D flat array and returns standard Python lists for JSON serialization."""
    flat_array = np.asarray(flat_array)
    reconstructed = []
    pointer = 0
    for shape in shapes:
        size = int(np.prod(shape))
        layer = flat_array[pointer : pointer + size].reshape(shape)
        reconstructed.append(layer.tolist())
        pointer += size
    return reconstructed


def initialize_population(
    n_individuals: int,
    input_size: int,
    output_size: int,
) -> Population:
    individuals = []

    for _ in range(n_individuals):
        individual = Individual()
        individual.genotype = make_random_weights(input_size, output_size)
        individuals.append(individual)

    return Population(individuals)

def evaluate(
    population: Population,
) -> Population:

    for individual in population.unevaluated:
        weights = individual.genotype
        individual.fitness = run_experiment(weights)

    return population

def parent_selection(population: Population) -> Population:

    shuffled = population.shuffle()
    for idx in range(0, len(shuffled) - 1, 2):
        ind_a = shuffled[idx]
        ind_b = shuffled[idx + 1]
        if ind_a.fitness_ is not None and ind_b.fitness_ is not None:
            if ind_a.fitness_ <= ind_b.fitness_:
                ind_a.tags = {"selected": True}
                ind_b.tags = {"selected": False}
            else:
                ind_a.tags = {"selected": False}
                ind_b.tags = {"selected": True}

    return shuffled

def log_progress(population: Population) -> Population:
    # Extract fitnesses of all currently alive individuals
    fitnesses = [ind.fitness_ for ind in population.alive if ind.fitness_ is not None]
    
    if fitnesses:
        best = min(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        worst = max(fitnesses)
        pop_size = len(population.alive)  # Get the current population size
        
        console.log(f"--- Generation Progress ---")
        console.log(f"Population Size : {pop_size}")
        console.log(f"Best Fitness    : {best:.4f} (closest to target)")
        console.log(f"Mean Fitness    : {mean:.4f}")
        console.log(f"Worst Fitness   : {worst:.4f}")
    
    return population


def uniform_crossover(
    genotype_a: list[npt.NDArray[np.float64]],
    genotype_b: list[npt.NDArray[np.float64]],
) -> tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:

    flat_a, shapes = _flatten_genotype(genotype_a)
    flat_b, _ = _flatten_genotype(genotype_b)

    mask = RNG.random(flat_a.shape) < 0.5
    child_flat_a = np.where(mask, flat_a, flat_b)
    child_flat_b = np.where(mask, flat_b, flat_a)

    child_a = _unflatten_genotype(child_flat_a, shapes)
    child_b = _unflatten_genotype(child_flat_b, shapes)

    return child_a, child_b

def n_point_crossover(
    genotype_a: list[npt.NDArray[np.float64]],
    genotype_b: list[npt.NDArray[np.float64]],
) -> tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
    
    flat_a, shapes = _flatten_genotype(genotype_a)
    flat_b, _ = _flatten_genotype(genotype_b)

    cut_points = sorted(
        RNG.choice(
            np.arange(1, len(flat_a)),
            size=N_CUTS,
            replace=False,
        ).tolist(),
    )

    boundaries = [0, *cut_points, len(flat_a)]

    child_flat_a = flat_a.copy()
    child_flat_b = flat_b.copy()

    for segment_index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
    ):
        if segment_index % 2 == 1:
            child_flat_a[start:end] = flat_b[start:end]
            child_flat_b[start:end] = flat_a[start:end]

    child_a = _unflatten_genotype(child_flat_a, shapes)
    child_b = _unflatten_genotype(child_flat_b, shapes)

    return child_a, child_b

def crossover(
    population: Population,
) -> Population:
    parents = population.where(lambda ind: bool(ind.tags.get("selected", False)))

    for idx in range(0, len(parents) - 1, 2):
        parent_a = parents[idx]
        parent_b = parents[idx + 1]

        if CROSSOVER == "uniform":
            child_genotype_a, child_genotype_b = uniform_crossover(
                parent_a.genotype,
                parent_b.genotype,
            )
        elif CROSSOVER == "n_point":
            child_genotype_a, child_genotype_b = n_point_crossover(
                parent_a.genotype,
                parent_b.genotype,
            )


        child_a = Individual()
        child_a.genotype = child_genotype_a
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = child_genotype_b
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])

    return population

def mutate(population: Population) -> Population:
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        
        flat_genotype, shapes = _flatten_genotype(ind.genotype)
        
        mutated_flat = FloatMutator.gaussian(
            flat_genotype,
            std=0.1,
            mutation_probability=0.2,
        )
        
        ind.genotype = _unflatten_genotype(mutated_flat, shapes)
        
        ind.requires_eval = True

    return population

def survivor_selection(population: Population) -> Population:
    ranked = population.alive.sort(
        sort="min",
        attribute="fitness_",
    )

    survivors = ranked[:TARGET_POPULATION]

    for individual in population:
        individual.alive = individual in survivors

    return population


def run_experiment(weights, mode: ViewerTypes = MODE) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos)
    output_size = model.nu

    # weights = make_random_weights(input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    # console.log(f"start  : {np.round(initial_position, 3)}")
    # console.log(f"end    : {np.round(final_position, 3)}")
    # console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    # console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""

    database_path = DATA / f"{CROSSOVER}_seed_{SEED}.db"
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    # run_experiment(MODE)

    population = initialize_population(
        n_individuals=STARTER_SIZE,
        input_size=input_size,
        output_size=output_size
    )
    
    population = evaluate(
        population=population,
    )

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover),
        EAOperation(mutate),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
        # EAOperation(log_progress),
    ]
    
    ea = EA(
        population,
        ops,
        num_steps=STEPS, 
        is_maximisation=False, 
        db_file_path=database_path,
        db_handling="delete",
        )
    ea.run()

    console.log("--- Results ---")
    console.log(f"best = {ea.get_solution('best', only_alive=False)}")
    console.log(f"median = {ea.get_solution('median', only_alive=False)}")
    console.log(f"worst = {ea.get_solution('worst', only_alive=False)}")


if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #
