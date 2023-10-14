import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import subprocess
from glob import glob
import bisect
from tqdm import tqdm
import astropy.units as u
import astropy.constants as c
import matplotlib as mpl

# Make it legible
mpl.rcParams['figure.dpi'] = 200

# Conversions from CLASS units
Hconv = (u.Mpc**-1*c.c).to(u.km/u.s/u.Mpc).value

parser = argparse.ArgumentParser(description="Visualize MontePython chain output in H(z)/(1+z) style",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Model parameters
parser.add_argument("classy_path", help="Where to find the class module (same directory as montepython spec)")
parser.add_argument("runs", nargs='+', action='append', help="MontePython run folders to use")
parser.add_argument("--burn", type=float, default=0.5, help="What fraction of the head of the chain to discard")
parser.add_argument("--thin", type=int, default=100, help="Select every nth link in the chain")
parser.add_argument("--fiducial", help="If provided, plot data normalized by this class ini file")
parser.add_argument("--ignoregit", action='store_true', help="Ignore discrepant git hashes (might just crash)")
parser.add_argument("--explore", type=int, help="Produce sliders for the MC explored parameters of this run (indexed from 0 in the order listed on the command line)")
parser.add_argument("--showdata", action='store_true', help="Include current measurements of H(z)")

args = parser.parse_args()

try:
    sys.path.insert(1, args.classy_path + "/python/build")
    import classy
    cosmo = classy.Class()
    
except ImportError:
    raise Exception("Unable to import the CLASS python module.  Check ur symlink")

# If we're exploring, make the model we explore last in the run order
if not args.explore is None:
    tmp = args.runs[0][-1]
    args.runs[0][-1] = args.runs[0][args.explore]
    args.runs[0][args.explore] = tmp
    args.explore = len(args.runs[0]) - 1
    
# KC 9/6/23
# MontePython must have some stuff for building Class() runs from the parameter file
# already.  I should be using that code.
# (I'm not readily finding it near .compute() call to class in sample.py...)

# Make a regexp for extracting the parameter names
hash_finder = re.compile(r'^#.*hash: ([a-f0-9]+)')
param_finder = re.compile(r'^data\.parameters\[\'([\w\^\{\}\*]+)\'\]\s*=\s*\[(.+)\]')
cosmo_finder = re.compile(r'\'cosmo\'')
arg_finder = re.compile(r'^data\.cosmo_arguments\[\'([\w\^\{\}\*]+)\'\]\s*=\s*\'?([^\'\"\s]+)\'?$')

def parseCLASSini(fname):
    # Parse the reference file and save the results in reference_cmb.cached
    tdict = {}
    for line in open(fname, 'r'):
        line = line.strip()

        # remove any further comments
        try:
            line = line[:line.index('#')]
        except ValueError:
            pass

        if len(line) == 0:
            continue

        pair = [x.strip() for x in line.split('=')]

        # Only make a setting if its not blank
        if len(pair[1]) > 0:
            try:
                tdict[pair[0]] = int(pair[1])
            except:
                try:
                    tdict[pair[0]] = float(pair[1])
                except:
                    tdict[pair[0]] = pair[1]
    return tdict
        
def loadRun(run):
    # Load the run params to get the order of the columns to
    # interpret
    print("Processing run folder:", run)
    
    with open(run + "/log.param", 'rt') as param_file:
        fit_parameter_ranges = {}
        fit_parameters = []
        fit_arguments = []
        column_names = []
        
        for line in param_file:

            # Retrieve the git hash and make sure
            # it agrees with the build sitting in the classy path
            if not args.ignoregit:
                match = hash_finder.match(line)
                if match:
                    git_hash = match.group(1)
                    found_hash = subprocess.getoutput('git -C %s rev-parse --verify HEAD' % args.classy_path)
                    if not git_hash == found_hash:
                        print("Discrepant hashes, run classy vs current classy:\nRun:\t\t%s\nCurrent:\t\t%s" % (git_hash, found_hash))
                        exit(1)
                    else:
                        print("Confirmed specified classy module has consistent hash:", git_hash)

                    continue
                
            # These are the column names, in the order they appear in the chains
            match = param_finder.match(line)
            if match:
                column_names.append(match.group(1))
                
                if cosmo_finder.search(line):

                    # Okay, this is one that we're going to search over in MCMC
                    fit_parameters.append(match.group(1))
                    print(match.group(1))
                    
                    # We also want the starting central value and minimum and maximum,
                    # in case we want to set up some sliders for interactive visualization
                    param_ranges = [x.strip() for x in match.group(2).split(',')]
                    fit_parameter_ranges[match.group(1)] = ({ 'central' : float(param_ranges[0]),
                                                              'min' : float(param_ranges[1]),
                                                              'max' : float(param_ranges[2]) })
                    
                continue

            # These are the arguments passed to class without search
            match = arg_finder.match(line)
            if match:
                try:
                    val = float(match.group(2))
                except ValueError:
                    val = match.group(2)

                # Append it
                fit_arguments.append((match.group(1), val))
        
    # We now have parameters sufficient to run a CLASS instance
    print(fit_parameters)
    print(fit_arguments)

    # Now, we find the chains, and load them into a big pandas
    chains = glob(run+"/*__*.txt")
    print(chains)

    dfs = []
    for chain in chains:
        dfs.append(pd.read_csv(chain, delim_whitespace=True, comment='#', names=(['id', '-loglike']+column_names), engine='python'))

    df = pd.concat(dfs)
    print(df)

    # Now we have all the chains, destroy the first burn
    df = df.iloc[int(args.burn*len(df)):]

    # Define thin target as how many to end up with
    stride = len(df) // args.thin

    try:
        # Now thin them
        df = df.iloc[::stride]
    except ValueError:
        print(f"[{run}]: Insufficient remaining posterior samples for requested burn and thin.\nSamples after burn: {len(df):d}\nTry reducing --burn and --thin")
        quit()
        
    # Return the frames
    return (fit_arguments, fit_parameters, fit_parameter_ranges, df)

def exploration_update_wrapper(val, fig, sliders, base_run_dict, fit_parameters, line):

    print("val", val)
    
    # Run it again with current slider positions
    bg = evaluate_bg(base_run_dict, fit_parameters, {param : sliders[param].val for param in fit_parameters})

    # Update the line
    line.set_ydata(bg['H [1/Mpc]']/(bg['z'] + 1) * Hconv)
    fig.canvas.draw_idle()
    
def evaluate_bg(base_run_dict, fit_parameters, row, maxz=3.0):
    # Get parameters for the specific run
    run_settings = {**base_run_dict, **{ field : row[field] for field in fit_parameters }}

    # Try to run a classy
    if cosmo.state:
        cosmo.struct_cleanup()

    cosmo.set(run_settings)
    cosmo.compute()

    # Get the background
    bg = cosmo.get_background()

    # Find where z < 3 stars in the array
    index = bisect.bisect(np.array(list(reversed(bg['z']))), maxz)

    # Truncate at the requested z
    for key,item in bg.items():
        bg[key] = bg[key][-index:]
    
    return bg

def add_datapoints(ax):

    from cycler import cycler
    
    datapts = []
    custom_cycler = (cycler(color=['red', 'green', 'yellow', 'blue', 'orange', 'pink', 'brown', 'cyan', 'purple']) + 
                     cycler(marker=['p', '3', '2', 'v', '>', 's', '*', 'd', 'X']))

    ax.set_prop_cycle(custom_cycler)
    
    # Add H0LiCOW 2019
    datapts.append(['Wong et al. 2019 (H0LiCOW)', (0.0), (73.3), (1.7), (--1.8)])

    # Add COSMICflows4
    datapts.append(['Tully et al. 2023 (Cosmicflows-4)', (0.0), (74.6), (3), (--3)])

    # SH0ES
    # 2306.00070
    datapts.append(['Murakami et al. 2023 (SH0ES)', (0.0), (73.29), (0.9), (--0.9)])

    # Cosmic chronometers
    
    # ref97 = ['97',
    #          (0.07, 0.12, 0.2, 0.28),
    #          (69.0, 68.6, 72.9, 88.8),
    #          (19.6, 26.2, 29.6, 36.6),
    #          (19.6, 26.2, 29.6, 36.6)]

    # ref98 = ['98',
    #          (0.09),
    #          (69.0),
    #          (12.0),
    #          (12.0)]

    # ref99 = ['99',
    #          (0.17, 0.27, 0.4, 0.9, 1.3, 1.43, 1.53, 1.75),
    #          (83.0, 77.0, 95.0, 117.0, 168.0, 177.0, 140.0, 202.0),
    #          (8.0, 14.0, 17.0, 23.0, 17.0, 18.0, 14.0, 40.0),
    #          (8.0, 14.0, 17.0, 23.0, 17.0, 18.0, 14.0, 40.0)]

    # ref100 = ['100',
    #           (0.1791, 0.1993, 0.3519, 0.5929, 0.6797, 0.7812, 0.8754, 1.037),
    #           (78.0, 78.0, 85.5, 107.0, 95.0, 96.5, 124.5, 133.5),
    #           (6.2, 6.9, 15.7, 15.5, 10.5, 12.5, 17.4, 17.6),
    #           (6.2, 6.9, 15.7, 15.5, 10.5, 12.5, 17.4, 17.6)]

    # ref101 = ['101',
    #           (0.3802, 0.4004, 0.4247, 0.4497, 0.4783),
    #           (86.2, 79.9, 90.4, 96.3, 83.8),
    #           (14.6, 11.4, 12.8, 14.4, 10.2),
    #           (14.6, 11.4, 12.8, 14.4, 10.2)]

    # ref102 = ['102',
    #           (0.47),
    #           (89.0),
    #           (49.6),
    #           (49.6)]

    # ref103 = ['103',
    #           (0.48, 0.88),
    #           (97.0, 90.0),
    #           (62.0, 40.0),
    #           (62.0, 40.0)]

    # ref104 = ['104',
    #           (0.75),
    #           (98.8),
    #           (33.6),
    #           (33.6)]

    # ref105 = ['105',
    #           (1.26),
    #           (135.0),
    #           (65.0),
    #           (65.0)]

    # ref106 = ['106',
    #           (1.363, 1.965),
    #           (160.0, 186.5),
    #           (33.8, 50.6),
    #           (33.8, 50.6)]

    # # Grab local scope here for eval calls
    # scope = locals()
            
    # for ref in [eval('ref%d' % n, scope) for n in range(97,107)]:
    #     datapts.append(ref)
        

    # BOSS DR12, 2017, Table 7
    datapts.append(['Alam et al. 2017 (BOSS DR12, BAO+FS)',
                    (0.38, 0.51, 0.61),
                    (81.5, 90.5, 97.3),
                    (2.6, 2.7, 2.9),
                    (--2.6, --2.7, --2.9)])

    # BOSS DR14 QSO, Table 4
    datapts.append(['Zhao et al. 2018 (BOSS DR14, QSO)',
                    (1.526),
                    (150.32),
                    (10.50),
                    (--10.50)])
    
    # BOSS DR14 Ly\alpha, Eqn. (35)
    datapts.append([r'Agathe et al. 2019 (BOSS DR14, Ly$\alpha$)',
                    (2.34),
                    (227),
                    (8),
                    (--8)])

    # # SDSS DR7
    # conv = c.c.to(u.km/u.s)
    # datapts.append([r'Ross et al. 2015 (SDSS DR7, galaxies)',
    #                 (0.15),
    #                 (conv/664),
    #                 (conv/25),
    #                 (--conv/25)])
                    
    for datapt in datapts:
        label, z_tuple, H_tuple, sigma_p_tuple, sigma_m_tuple = datapt
        yerr = np.row_stack([np.asarray(x) for x in (sigma_p_tuple, sigma_m_tuple)])
        z_tuple = np.asarray(z_tuple)
        H_tuple = np.asarray(H_tuple)
        ax.errorbar(z_tuple, H_tuple/(1+z_tuple), yerr=yerr/(1+z_tuple), lw=1, label=label, capsize=5, ls='none')

    
# Get fiducial if its there
ref_H = None

if args.fiducial:
    params = parseCLASSini(args.fiducial)
    
    cosmo.set(params)
    cosmo.compute()

    bg = cosmo.get_background()
    ref_H = interp1d(bg['z'], bg['H [1/Mpc]'] * Hconv)
    cosmo = classy.Class()
    
# Go through and do this for all runs
colors = ['blue', 'red', 'green']

main_ax = plt.gca()
main_fig = plt.gcf()

for N, run in enumerate(args.runs[0]):

    # Load the run, burning and thinning
    fit_arguments, fit_parameters, fit_parameter_ranges, df = loadRun(run)

    # Now we can start calling CLASS to get expansion histories to plot
    # I'm sorry we do this row by row, no other way :/
    base_run_dict = { field : value for field,value in fit_arguments }

    cosmo = classy.Class()
    
    label = run.split('/')[-2]
    
    for index,row in (pbar := tqdm(df.iterrows(), total=len(df))):

        try:
            bg = evaluate_bg(base_run_dict, fit_parameters, row)

            H = bg['H [1/Mpc]'] * Hconv
            z = bg['z']

            if ref_H is None:
                main_ax.plot(z, H/(1+z), alpha=0.1, color=colors[N], label=label, linewidth=0.5)
            else:
                main_ax.plot(z, H/ref_H(z), alpha=0.1, color=colors[N], label=label, linewidth=0.5)

            if label:
                label = None
                
        except classy.CosmoComputationError as e:
            print("Skipping...\n", row, "you might have version skew between your CLASS and this run. (--ignoregit flag is)", args.ignoregit)
            
    # Add sliders for this run
    if N == args.explore:
        from matplotlib.widgets import Slider, Button

        # Draw the first model line
        # I should be able to pass central values in a dict, not a df
        bg = evaluate_bg(base_run_dict, fit_parameters, {param : fit_parameter_ranges[param]['central'] for param in fit_parameters})
        explore_line, = main_ax.plot(bg['z'], bg['H [1/Mpc]']/(1+bg['z']) * Hconv, lw=2, color='black', linestyle=':', label="Exploring "+run.split('/')[-2])

        # Make room for the parameter knobs
        right_edge = 0.7
        bottom_edge = 0.1
        main_fig.subplots_adjust(right=right_edge, bottom=bottom_edge)
        
        # Make a vertically oriented sliders to control all the fit parameters
        # with ranges set by what the MC was run with (because these are all
        # the values the run was capable of searching, so it doesn make sense
        # to visualize ones outside these ranges)
        sliders = {}
        
        for j,param in enumerate(fit_parameters):
            knobax = main_fig.add_axes([right_edge + 0.05 + j*0.05, bottom_edge, 0.0225, 0.8])
            slider = Slider(
                ax=knobax,
                label=param,
                valmin=fit_parameter_ranges[param]['min'],
                valmax=fit_parameter_ranges[param]['max'],
                valinit=fit_parameter_ranges[param]['central'],
                orientation="vertical"
            )

            # Register a changing dakine
            # This will use evaluation-time versions of these variables
            # so it will run on the last one.
            #
            # So just juggle the run order
            slider.on_changed(lambda val : exploration_update_wrapper(val, main_fig, sliders, base_run_dict, fit_parameters, explore_line))

            # Store it in a dict
            sliders[param] = slider

main_ax.set_ylim((52,85))

main_ax.set_xlabel(r'Redshift $z$')
main_ax.set_ylabel(r'Expansion velocity km s$^{-1}$ Mpc$^{-1}$')

if args.showdata:
    add_datapoints(main_ax)
    
main_ax.legend()

plt.show()


