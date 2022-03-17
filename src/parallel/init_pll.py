
# This session number should be set to be the same in all 3 scripts
SES_NUM = 2
filename = f"pll_log_{SES_NUM}.csv"
file = open(filename, "w")
# Save the file headers
file.write("N, t_build, t_total\n")
file.close()
print("Starting Parallel Simulations for SES_NUM:", SES_NUM)
