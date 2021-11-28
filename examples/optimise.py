import argparse
import pandas as pd
import numpy as np

from protein_design.specifications import Monomer, SPECIFICATIONS
from protein_design.specifications import get_buff_total_energy
from protein_design.specifications import build_model

from isambard.optimisation.evo_optimizers import Parameter
import isambard.optimisation.evo_optimizers as ev_opts
import isambard


def parse_parameters_csv(parameters_csv):
    df = pd.read_csv(parameters_csv)
    parameters = []
    for p_type, name, mean, var in df.values:
        if p_type == 'dynamic':
            parameters.append(Parameter.dynamic(name, eval(mean), eval(var)))
        elif p_type == 'static':
            parameters.append(Parameter.static(name, eval(mean)))
        else:
            raise ValueError(f"Unknown parameter type '{p_type}'!")
    return parameters


def parse_sequences_csv(sequence_csv):
    df = pd.read_csv(sequence_csv)
    return {ID: list(df.iloc[i, 1:].values) for i, ID in enumerate(df.iloc[:, 0].values)}


def get_coordinate_system_scapTet(monomer):
    '''Caclulates the local coordinate system of a monomer of scapTet
    Local coordinate system contains 3 orthogonal unit vectors representing the orientation of the monomer.

    Args:
        monomer (isambard.ampal.Assembly): Assembly object containing the scapTet

    Returns:
        A tuple of 3 (np.ndarray) vectors (v24, v13, z) where
            v24 extends from helix 2 to helix 4
            v13 extends from helix 1 to helix 3
            z = v24 cross v13
    '''
    start = np.sum([monomer[0][1:13].backbone.centre_of_mass,
                    monomer[0][44:56].backbone.centre_of_mass,
                    monomer[0][64:76].backbone.centre_of_mass,
                    monomer[0][108:122].backbone.centre_of_mass], axis=0)/4.
    end = np.sum([monomer[0][13:26].backbone.centre_of_mass,
                  monomer[0][32:44].backbone.centre_of_mass,
                  monomer[0][76:89].backbone.centre_of_mass,
                  monomer[0][95:108].backbone.centre_of_mass], axis=0)/4.
    z = end-start
    z /= np.linalg.norm(z)

    v13 = monomer[0][1:26].backbone.centre_of_mass - \
        monomer[0][64:89].backbone.centre_of_mass
    v24 = monomer[0][32:56].backbone.centre_of_mass - \
        monomer[0][95:122].backbone.centre_of_mass
    v13 -= np.dot(v13, z)*z
    v24 -= np.dot(v24, z)*z
    v13 /= np.linalg.norm(v13)
    v24 /= np.linalg.norm(v24)

    return v24, v13, z


def opt(args):
    print("[loading sequence_csv]")
    all_sequences = parse_sequences_csv(args.sequences_csv)
    print("[loading scapTet_pdb]")
    scapTet = isambard.ampal.load_pdb(args.scapTet_pdb)
    monomer = Monomer(
        scapTet, get_coordinate_system_func=get_coordinate_system_scapTet)
    print("[loading parameters_csv]")
    parameters = [Parameter.static('monomer', monomer)]
    parameters.extend(parse_parameters_csv(args.parameters_csv))

    for seq_id, sequences in all_sequences.items():
        print(
            f"[optimising seq_id='{seq_id}' using GA with {args.cores} cores ...]")
        print(f"sequences : {sequences}")
        opt_ga = ev_opts.GA(SPECIFICATIONS[args.specification], sequences, parameters,
                            eval_fn=get_buff_total_energy,
                            build_fn=build_model)

        opt_ga.run_opt(pop_size=args.population,
                       generations=args.generations,
                       cores=args.cores)

        best_dimer = opt_ga.best_model
        best_idx = opt_ga.halloffame[0]
        final_params = opt_ga.parse_individual(best_idx)

        print(f"[saving best model for '{seq_id}']")
        with open(f'best_model_{seq_id}.pdb', 'w') as f:
            f.write(best_dimer.pdb)

        print(f"[writing best parameters for '{seq_id}']")
        with open(f'best_params_{seq_id}.txt', 'w') as f:
            f.write(final_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("specification", default='Dimer',
                        choices=list(SPECIFICATIONS.keys()))
    parser.add_argument("scapTet_pdb")
    parser.add_argument("sequences_csv")
    parser.add_argument("parameters_csv")
    parser.add_argument("population", default=500, type=int)
    parser.add_argument("generations", default=5, type=int)
    parser.add_argument("cores", default=8, type=int)
    args = parser.parse_args()
    opt(args)
