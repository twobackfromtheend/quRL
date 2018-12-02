import qutip.logging_utils as logging
import numpy as np
from qutip import *
import datetime
import qutip.control.pulseoptim as cpo
import matplotlib.pyplot as plt

logger = logging.get_logger()
log_level = logging.INFO

H_d = -sigmaz() / 2  # Factor of 2 for Spin matrix
H_c = [-sigmax() / 2]

initial_state = (-sigmaz() + 2 * sigmax()).groundstate()[1]
target_state = (-sigmaz() - 2 * sigmax()).groundstate()[1]

example_name = 'CRAB_3'
p_type = 'SINE' #Only for GRAPE

n_ts = 60  # No. of timesteps
evo_time = 3  # Total protocol duration

# fid = 0.33129935776847313 # but this is squared
# fid = 0.57558609935  # unsquared

# fid_err_targ = 1 - fid  # 0.42441390064
fid_err_targ = 0

f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)  # Saving to file

# "GRAPE"
# result = cpo.optimize_pulse_unitary(H_d, H_c, initial_state, target_state, n_ts, evo_time,
#                                     fid_err_targ=fid_err_targ, amp_lbound=-4, amp_ubound=4,
#                                     log_level=log_level,
#                                     out_file_ext=f_ext, gen_stats=True
#                                     , init_pulse_type='SINE'
#                                     , pulse_scaling=4.0)

"CRAB"
result = cpo.opt_pulse_crab_unitary(H_d, H_c, initial_state, target_state, n_ts, evo_time,
                                    fid_err_targ=fid_err_targ, amp_lbound=-4, amp_ubound=4,
                                    log_level=log_level,
                                    out_file_ext=f_ext, gen_stats=True,
                                    init_coeff_scaling=4)

result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
    datetime.timedelta(seconds=result.wall_time)))

"Plotting Control Amplitudes"
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial Control amps")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Amplitudes")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
plt.tight_layout()
plt.show()
