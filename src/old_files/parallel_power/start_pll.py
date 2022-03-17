
# This session number should be set to be the same in all 3 scripts
SES_NUM = 7
filename = f"power_log_{SES_NUM}.csv"
file = open(filename, "w")
# Save the file headers
file.write("N, NR, RS, PMAX, PM POS, NOTES\n")
file.close()
print("Starting Parallel Calculations")
