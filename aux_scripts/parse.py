"""Parse txt log files
Not necessary now that I'm saving the RMSR values straight to csv
"""
import argparse
import numpy as np
import parse_util as pu
#import pandas as pd # readcsv

# Parse the arguments
desc = "Parse a log file, extracting currents or RMSR values"
parser = argparse.ArgumentParser(description=desc)
fh = "Provide the filename of a log.txt file"
parser.add_argument("filename", nargs="+", help=fh)
th = "Which type of log? (FP, train, trst)"
parser.add_argument("-t", "--type", help=th)
args = parser.parse_args()

mode = "FP" # "train"
#filename = "./train_66/log_66.txt"
#filename = "./69_train/log_69NK.txt"
#filename = "./70_train/log_70trf.txt"
filenames = args.filename
if args.type is not None:
  mode = args.type

outfile = "parsed.csv"

# Pull out the input & output currents from a training log
def IO_currents(filename):

  print(f"Parsing {mode} log ", filename)

  lfile = open(filename, "r")
  lines = lfile.readlines()
  line_list = []

  if mode == "trst":
    i_start_str = "The output currents"
    reading_currents = False
    i_str = ""
    ii = 0
    old_format = False

    if old_format:
      for line in lines:
        if i_start_str in line:
          reading_currents = True
        elif reading_currents:
          i_str += line #+ "\n"
          if "]]" in line:
            #print(47, i_str)
            #print(48, pu.str2np(i_str))
            pRMSRM, pRMSRR = pu.str_RMSR(i_str)
            line_list.append([ii, pRMSRM, pRMSRR])
            reading_currents = False
            i_str = ""
            ii += 1
          print(55, line_list)
    else:
      i_arr = []
      ii = 0
      for line in lines:
        if i_start_str in line:
          reading_currents = True
        elif reading_currents:
          if "[" not in line:
            reading_currents = False
            # Done reading the currents
            i_arr = np.vstack(i_arr)
            pRMSRM = pu.Is2RMSR(i_arr, mode="max", msg=None)
            pRMSRR = pu.Is2RMSR(i_arr, mode="rel", msg=None)
            line_list.append([filename, ii, pRMSRM, pRMSRR])
            ii += 1
            i_arr = []
          else:
            i_str = line.split(":")[1].strip("[] \n")
            ii_arr = np.fromstring(i_str, sep=" ") 
            i_arr.append( ii_arr )

  elif mode == "FP":
    in_str = "Sending in this input:"
    out_str = "The output currents were:"
    out_cs_2d = []

    in_i = "0bXX"
    in_n = 0
    for line in lines:
      if in_str in line:
        in_i = line.split(in_str)[1].strip()
        in_n += 1
      if out_str in line:
        cs_str = line.split(out_str)[1]
        cs_arr = cs_str.strip("[] \n").split(" ")
        while "" in cs_arr:
          cs_arr.remove("")
        out_cs = np.array(cs_arr, dtype=np.float64)
        line_list.append([in_n, in_i, out_cs])
        out_cs_2d.append(out_cs)
    out_currents = np.vstack(out_cs_2d)
    # Calculate the RMSR
    #IO_Ns = pd.read_csv("93_trsts/IO_XOR.csv", skipinitialspace=True).to_numpy()
    pu.Is2RMSR(out_currents, msg="(max)", mode="max")
    pu.Is2RMSR(out_currents, msg="(rel)", mode="rel")
  elif mode == "train":
    epoch = 0

    out_str = "Current leaving output pins:"
    in_str = "Current entering input pins:"
    epoch_done_str = "Output:"

    for line in lines:
      if epoch_done_str in line:
        # Save
        line_list.append([epoch, out_cs, in_cs])
        epoch = epoch + 1
      elif out_str in line:
        cs_str = line.split(out_str)[1]
        cs_arr = cs_str.strip("[] \n").split(" ")
        while "" in cs_arr:
          cs_arr.remove("")
        out_cs = np.array(cs_arr, dtype=np.float64)
      elif in_str in line:
        cs_str = line.split(in_str)[1]
        cs_arr = cs_str.strip("[] \n").split(" ")
        while "" in cs_arr:
          cs_arr.remove("")
        in_cs = np.array(cs_arr, dtype=np.float64)
  lfile.close()

  dlm = ","
  ofile = open(outfile, "a")
  for line_i in line_list:
    for entry in line_i:
      if type(entry) in (int, float, str, np.float64):
        ofile.write(str(entry))
        ofile.write(dlm)
      else:
        # Assume it's an iterable
        for subentry in entry:
          ofile.write(str(subentry))
          ofile.write(dlm)
    ofile.write("\n")
  ofile.close()

for filename in filenames:
  IO_currents(filename)

