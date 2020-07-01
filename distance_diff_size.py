from PIL import Image
from SSIM_PIL import compare_ssim
from PIL import Image
import requests
from io import BytesIO
from sklearn import metrics
import numpy as np
#import cv2
#from SSIM_PIL import compare_ssim
from PIL import Image

def distance_diff_size_plot(images_sentinel,images_naip):

    #right location experiment

    MSE_global_r = []
    SSIM_global_r = []

    for i in range(0,20):
      #print(i)
      drone_url = images_sentinel[i]
      planet_url = images_naip[i]
      location = "TRUE"
      #response = requests.get(drone_url)
      drone_img = Image.open(drone_url)
      planet_img = Image.open(planet_url)
      planet_img = planet_img.convert('RGB')
      drone_img = drone_img.resize((512,512))
      base_planet_img = planet_img.resize((512, 512))

      x_1 = []
      for i in range(1,43):
        num = 12*i
        if (num % 2) != 0:
          num = num+1
        x_1.append(num)
        if i == 42:
          x_1.append(512)
      #print(x_1)

      if location == "TRUE":
        MSE_tt = []
        SSIM_tt = []
        for i in range(0,43):
          if i in range(0,42):
            #print(i)
            width = x_1[i]/2
            #print(width)
            cropped_drone_img = drone_img.crop((256 - width, 256 - width, 256 + width, 256 + width))
            #print("size",cropped_drone_img.size)
            cropped_drone_img = drone_img.crop((256 - width, 256 - width, 256 + width, 256 + width)).resize((512, 512))
            cropped_planet_img = base_planet_img.crop((256 - width, 256 - width, 256 + width, 256 + width)).resize((512, 512))
            np_drone = np.array(cropped_drone_img)
            np_planet = np.array(cropped_planet_img)
            MSE = metrics.mean_squared_error(np_drone.flatten(), np_planet.flatten())
            MSE_tt.append(MSE)
            SSIM = compare_ssim(cropped_drone_img,cropped_planet_img)
            SSIM_tt.append(SSIM)
          else:
            #print(i, "after 500")
            if i == 43:
              cropped_drone_img = cropped_drone_img.resize((512, 512))
              cropped_planet_img = base_planet_img.resize((512, 512))
            cropped_drone_img = cropped_drone_img.resize((i*12, i*12))
            cropped_planet_img = base_planet_img.resize((i*12, i*12))
            #print(cropped_planet_img.size)
            np_drone = np.array(cropped_drone_img)
            np_planet = np.array(cropped_planet_img)
            MSE = metrics.mean_squared_error(np_drone.flatten(), np_planet.flatten())
            MSE_tt.append(MSE)
            SSIM = compare_ssim(cropped_drone_img,cropped_planet_img)
            SSIM_tt.append(SSIM)
        #print(len(MSE_global_r))
        SSIM_global_r.append(SSIM_tt)
        MSE_global_r.append(MSE_tt)

    #wrong location experiment

    import random

    location = "WRONG"

    MSE_global_wl = []
    SSIM_global_wl = []

    def rotate(l, n):
      return l[n:] + l[:n]

    for i in range(0,20):
      drone_url = images_sentinel[i]
      images_naip_rot = rotate(images_naip, 2)
      planet_url = images_naip_rot[i]
      drone_img = Image.open(drone_url)
      planet_img = Image.open(planet_url)
      planet_img = planet_img.convert('RGB')
      drone_img = drone_img.resize((512,512))
      base_planet_img = planet_img.resize((512, 512))

      x_1 = []
      for i in range(1,43):
        num = 12*i
        if (num % 2) != 0:
          num = num+1
        x_1.append(num)
        if i == 42:
          x_1.append(512)

      if location == "WRONG":
        MSE_tt = []
        SSIM_tt = []
        for i in range(0,43):
          if i in range(0,42):
            width = x_1[i]/2
            cropped_drone_img = drone_img.crop((256 - width, 256 - width, 256 + width, 256 + width))
            cropped_drone_img = drone_img.crop((256 - width, 256 - width, 256 + width, 256 + width)).resize((512, 512))
            cropped_planet_img = base_planet_img.crop((256 - width, 256 - width, 256 + width, 256 + width)).resize((512, 512))
            np_drone = np.array(cropped_drone_img)
            np_planet = np.array(cropped_planet_img)
            MSE = metrics.mean_squared_error(np_drone.flatten(), np_planet.flatten())
            MSE_tt.append(MSE)
            SSIM = compare_ssim(cropped_drone_img,cropped_planet_img)
            SSIM_tt.append(SSIM)
          else:
            if i == 43:
              cropped_drone_img = cropped_drone_img.resize((512, 512))
              cropped_planet_img = base_planet_img.resize((512, 512))
            cropped_drone_img = cropped_drone_img.resize((i*12, i*12))
            cropped_planet_img = base_planet_img.resize((i*12, i*12))
            np_drone = np.array(cropped_drone_img)
            np_planet = np.array(cropped_planet_img)
            MSE = metrics.mean_squared_error(np_drone.flatten(), np_planet.flatten())
            MSE_tt.append(MSE)
            SSIM = compare_ssim(cropped_drone_img,cropped_planet_img)
            SSIM_tt.append(SSIM)
        MSE_global_wl.append(MSE_tt)
        SSIM_global_wl.append(SSIM_tt)

    import pandas as pd
    x_2 = [i*i for i in x_1]
    df_r = pd.DataFrame(MSE_global_r, index =["image 1", "image 2", "image 3", "image 4", "image 5", "image 6","image 7", "image 8", "image 9","image 10","image 11", "image 12", "image 13", "image 14", "image 15", "image 16","image 17", "image 18", "image 19","image 20"], columns =x_2)
    tmp_df_r = df_r

    df_wl = pd.DataFrame(MSE_global_wl, index =["image 1", "image 2", "image 3", "image 4", "image 5", "image 6","image 7", "image 8", "image 9","image 10","image 11", "image 12", "image 13", "image 14", "image 15", "image 16","image 17", "image 18", "image 19","image 20"], columns =x_2)
    tmp_df_wl = df_wl

    A_r = []

    for k in tmp_df_r.keys():
      row_df = tmp_df_r[k]
      #row_df_wt = tmp_df_wt[k]
      row_df_wl = tmp_df_wl[k]
      for row in row_df.index:
        a = [row, float(row_df[row]), float(k), "rl"]
        #b = [row, float(row_df_wt[row]), float(k), "wt-rl"]
        c = [row, float(row_df_wl[row]), float(k), "wl"]
        A_r += [a]
        #A_r += [b]
        A_r += [c]

    new_pd_r = pd.DataFrame(A_r, columns=["Image", "Distance", "Area", "Experiment"])

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt = sns.lineplot(x="Area", y="Distance",
                 hue="Experiment",
                 data=new_pd_r, palette=["g","r"])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.set(xlabel="Area in $m^2$", ylabel='MSE')
    #sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    #sns.ticklabel_format(self,axis='both', style='', scilimits=None, useOffset=None, useLocale=None, useMathText=None)
