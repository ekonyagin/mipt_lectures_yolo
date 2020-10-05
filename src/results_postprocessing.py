#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from config import calibration_cfg

#df = pd.DataFrame({"height":heights,
#                    "width":widths,
#                    "areas":areas}).to_excel("out_res.xls")

def make_measurement_results(widths, heights, rotations, areas=None):
	widths = np.array(widths)
	heights = np.array(heights)
	rotations = np.array(rotations)
	
	if areas == None:
	 	areas = widths * heights
	else:
		areas = np.array(areas)

	if (len(areas) !=len(widths) or (len(widths) != len(heights))):
		raise ValueError("Length of widths must match\
		 length of areas and heights!")
	plt.figure(figsize=(12,8))
	plt.grid()
	plt.plot(areas, label='areas')
	plt.legend()
	plt.savefig('results/res_area.png')

	plt.figure(figsize=(12,8))
	plt.grid()
	plt.plot(rotations, label='rot')
	plt.legend()
	plt.savefig('results/res_rot.png')

	plt.figure(figsize=(12,8))
	plt.grid()
	plt.plot(widths,label='widths')
	plt.plot(heights, label='heigths')
	plt.legend()
	plt.savefig('results/res_h_w.png')

	plt.figure(figsize=(12,8))
	plt.grid()
	plt.plot(widths*(calibration_cfg.CALIBRATIONS["width_coef"]) + \
		calibration_cfg.CALIBRATIONS["width_offset"],label='dist')
	plt.legend()
	plt.savefig('results/res_dist.png')
