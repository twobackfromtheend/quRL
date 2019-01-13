import qutip.logging_utils as logging
import numpy as np
from qutip import *
import datetime
import qutip.control.pulseoptim as cpo
import matplotlib.pyplot as plt
import csv
from investigation.hamiltonian_generator import hamiltonian_generator


logger = logging.get_logger()
# log_level = logging.INFO

"Multi Qubit"
initial_state = hamiltonian_generator(-2)[0].groundstate()[1]
target_state = hamiltonian_generator(2)[0].groundstate()[1]


H_d = hamiltonian_generator(1)[1]
H_c = [hamiltonian_generator(1)[2]]


""


"Single Qubit"

# H_d = -sigmaz() / 2  # Factor of 2 for Spin matrix
# H_c = [-sigmax() / 2]
#
# initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
# target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]

""


max_fidelity_list = []
example_name = 'CRAB_3'
p_type = 'SINE'  # Only for GRAPE


dt1 = datetime.datetime.now()

t = np.arange(0.1, 3.3, 0.1)

for i in t:
    t_n = int(round(i / 0.05))

    n_ts = t_n  # No. of timesteps
    evo_time = i  # Total protocol duration

    # "Optimal Control involves minimising fid_err_targ = 1 - fidelity"
    # fid = 0.33129935776847313 # but this is squared
    # fid = 0.57558609935  # unsquared

    # fid_err_targ = 1 - fid  # 0.42441390064
    # fid_err_targ = 0

    # f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)  # Saving to file

# n_ts = 60
# evo_time = 3.0

    "GRAPE"
    result = cpo.optimize_pulse_unitary(H_d, H_c, initial_state, target_state, n_ts, evo_time,
                                        amp_lbound=-4, amp_ubound=4,
                                        # log_level=log_level,
                                        # out_file_ext=f_ext,
                                        gen_stats=True
                                        , init_pulse_type='DEF'
                                        , pulse_scaling=4.0)

    # "CRAB"
    # result = cpo.opt_pulse_crab_unitary(H_d, H_c, initial_state, target_state, n_ts, evo_time,
    #                                     # fid_err_targ=fid_err_targ,
    #                                     amp_lbound=-4, amp_ubound=4,
    #                                     # log_level=log_level,
    #                                     # out_file_ext=f_ext,
    #                                     gen_stats=True,
    #                                     init_coeff_scaling=4)

    # result.stats.report()
    # print("Final evolution\n{}\n".format(result.evo_full_final))
    final_fidelity = (fidelity(target_state, result.evo_full_final)) ** 2
    max_fidelity_list.append(final_fidelity)

# print("********* Summary *****************")
# print("Final fidelity error {}".format(result.fid_err))
# print("Final gradient normal {}".format(result.grad_norm_final))
# print("Terminated due to {}".format(result.termination_reason))
# print("Number of iterations {}".format(result.num_iter))
# print("Final Fidelity: ", final_fidelity)
# print("Completed in {} HH:MM:SS.US".format(
#     datetime.timedelta(seconds=result.wall_time)))

# "Plotting Control Amplitudes"
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(2, 1, 1)
# ax1.set_title("Initial Control amps")
# ax1.set_ylabel("Control amplitude")
# ax1.step(result.time,
#          np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
#          where='post')
#
# ax2 = fig1.add_subplot(2, 1, 2)
# ax2.set_title("Optimised Control Amplitudes")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Control amplitude")
# ax2.step(result.time,
#          np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
#          where='post')
# plt.tight_layout()
# plt.show()

dt2 = datetime.datetime.now()

print("Time: ", dt2 - dt1)

print(max_fidelity_list)

plt.plot(t, max_fidelity_list)
plt.xlabel('T')
plt.ylabel('Fidelity')
plt.grid()
# plt.xlim(0, 3.0)
# plt.ylim(0, 1)
plt.show()
