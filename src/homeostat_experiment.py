import sys
# relative path to folder which contains the Sandbox module
sys.path.insert(1, './Sandbox_v1_2')
from Sandbox import *
import copy as cp
from tqdm import *
from Homeostat_internals import *
import os
import time


def main(mutation_rate=0.01, crossover_rate=0.1,run=0,optim='PSO'):

	

	num_units = 3
	mutation_rate = mutation_rate
	crossover_rate = crossover_rate

	fig_size = (16, 10)
	timestamp = time.strftime('%Y%m%d_%H%M%S')
	if optim == 'PSO':
		run_folder = f'PSO_{run}_run_{num_units}_units_{mutation_rate}_c1_{crossover_rate}_c2_{timestamp}'
	else:
		run_folder = f'GA_{run}_run_{num_units}_units_{mutation_rate}_mr_{crossover_rate}_cr_{timestamp}'
	output_folder = os.path.join('output_plots', run_folder)
	os.makedirs(output_folder, exist_ok=True)

	homie = Homeostat(n_units=num_units, upper_viability=1, lower_viability=-1, upper_limit=10, lower_limit=-10, adapt_fun=random_val, adapt_enabled=True, test_interval=20,adaptation_cooldown=20,mutation_rate=mutation_rate, crossover_rate=crossover_rate)

	for u in homie.units:
		u.randomise_params()


	i = 0 
	t = 0
	ts = [t]
	dt = 0.01
	duration = 100

	while t < duration:
		homie.step(dt)

		t += dt
		ts.append(t)
		i += 1



	# PLOT 1: plot system state over time, showing when weights change
	plt.figure(figsize=fig_size)
	for i, unit in enumerate(homie.units):
		plt.plot(ts, unit.thetas, label='Unit '+str(i))

	# plot upper and lower viability boundaries
	plt.plot([ts[0], ts[-1]],[homie.units[0].upper_viability, homie.units[0].upper_viability], 'r--', label='Viability\nboundaries')
	plt.plot([ts[0], ts[-1]],[homie.units[0].lower_viability, homie.units[0].lower_viability], 'r--')

	# plot upper and lower hard limits
	plt.plot([ts[0], ts[-1]],[homie.units[0].upper_limit, homie.units[0].upper_limit], 'g--', label='Hard limits')
	plt.plot([ts[0], ts[-1]],[homie.units[0].lower_limit, homie.units[0].lower_limit], 'g--')


	# plot times when units start testing new weights
	for i, t_t in enumerate(homie.units[0].test_times):
		if i:
			l = None
		else:
			l = 'Weights begin to change'
		plt.plot([t_t, t_t], [homie.units[0].lower_limit, homie.units[0].upper_limit], 'b--', label=l, linewidth=3)

	plt.xlabel('t')
	plt.ylabel(r'$\theta$')
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.title("System state over time")


	plt.savefig(os.path.join(output_folder, 'plot1_system_state_with_weight_changes.png'))

	# PLOT 2: plot system state over time, *without* showing when weights change
	plt.figure(figsize=fig_size)
	for i, unit in enumerate(homie.units):
		plt.plot(ts, unit.thetas, label='Unit '+str(i))

	# plot upper and lower hard limits
	plt.plot([ts[0], ts[-1]],[homie.units[0].upper_viability, homie.units[0].upper_viability], 'r--', label='Viability\nboundaries')
	plt.plot([ts[0], ts[-1]],[homie.units[0].lower_viability, homie.units[0].lower_viability], 'r--')

	# plot upper and lower hard limits
	plt.plot([ts[0], ts[-1]],[homie.units[0].upper_limit, homie.units[0].upper_limit], 'g--', label='Hard limits')
	plt.plot([ts[0], ts[-1]],[homie.units[0].lower_limit, homie.units[0].lower_limit], 'g--')

	plt.xlabel('t')
	plt.ylabel(r'$\theta$')
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
	plt.title("System state over time")

	plt.savefig(os.path.join(output_folder, 'plot2_system_state_without_weight_changes.png'))

	# PLOT 3: plot all Homeostat unit weights over time
	plt.figure(figsize=fig_size)
	for i, unit in enumerate(homie.units):
		plt.plot(ts, unit.weights_hist, label='Unit ' + str(i) + ': weight')
	plt.title('Homeostat unit weights')
	plt.xlabel('t')
	plt.ylabel('Weights')
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

	plt.savefig(os.path.join(output_folder, 'plot3_homeostat_unit_weights.png'))

	# # PLOT 4: plot when all Homeostat units are adapting
	# plt.figure(figsize=fig_size)
	# for i, unit in enumerate(homie.units):
	# 	plt.plot(ts, unit.testing_hist, label='Unit '+str(i))
	# plt.xlabel('t')
	# plt.ylabel('Adapting')
	# plt.title('Units in process of adapting')
	# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

	# plt.savefig(os.path.join(output_folder, 'plot4_units_adapting.png'))


if __name__ == '__main__':
	mr = 1
	cr = 2

	runs = 5


	for run in range(runs):
		# for mr in mrs:
		# 	for cr in crs:
		# 		print(f'Params: mr = {mr}, cr = {cr}, run = {run}')
		main(mutation_rate=mr, crossover_rate=cr,run=run)


	# main(mutation_rate=1,crossover_rate=2, run=0)



