import os
from SimpleITK.SimpleITK import And
import numpy as np
import scipy
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd

from utils import  GetSlicesArray,StapleGun, GetAreaArray, SlicesSkip, StapleGun4, StapleGunPLIC, UnpackNpz, UnpackSlices, NewRemoveCoronaRadiataSlices, IdSkip4, DiceCaster, AverageDice, GetSliceOverlap4, SaveStapleData

def main():
    print("started main")
    adrian_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/CR_slices_adrian.npz'
    sacha_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/CR_slices_sacha.npz'
    paul_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/CR_slices_paul.npz'
    h_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/CR_slices_h.npz'

    slices_adrian, ids_adrian = UnpackSlices(adrian_path)
    slices_paul, ids_paul = UnpackSlices(paul_path)
    slices_sacha, ids_sacha = UnpackSlices(sacha_path)
    slices_h, ids_h = UnpackSlices(h_path)
    """
    adrian_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/PLIC_mask_adrian.npz'
    paul_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/PLIC_mask_paul.npz'
    sacha_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/PLIC_mask_sacha.npz'
    h_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/PLIC_mask_h.npz'
    """
    adrian_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/adrian_masks.npz'
    paul_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/paul_masks.npz'
    sacha_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/sacha_masks.npz'
    h_PLIC_path = '/Users/oliviamurray/Documents/PhD/MISTIE/mask_data/h_masks.npz'



    PLIC_masks_adrian, PLIC_ids_adrian = UnpackNpz(adrian_PLIC_path)
    print("ad")
    PLIC_masks_paul, PLIC_ids_paul = UnpackNpz(paul_PLIC_path)
    print("paul")
    PLIC_masks_sacha, PLIC_ids_sacha = UnpackNpz(sacha_PLIC_path)
    print("sacha")
    PLIC_masks_h, PLIC_ids_h = UnpackNpz(h_PLIC_path)
    print("h")
    #ct_path = '/Users/olivia/Documents/PhD/MISTIE/mask_data/CTscans.npz'
    #ct_scans, ct_ids = UnpackNpz(ct_path)
    print("unpacked")

    # remove patients not in all 4 arrays 

    PLIC_new_mask_paul,PLIC_new_id_paul,PLIC_new_mask_adrian,PLIC_new_id_adrian,PLIC_new_mask_sacha,PLIC_new_id_sacha, PLIC_new_mask_h, PLIC_new_id_h = IdSkip4(PLIC_ids_paul, PLIC_ids_adrian, PLIC_ids_sacha, PLIC_ids_h, PLIC_masks_paul, PLIC_masks_adrian, PLIC_masks_sacha, PLIC_masks_h)
    new_slices_adrian,new_slices_paul, new_slices_sacha, new_slices_h = SlicesSkip(PLIC_new_id_adrian, slices_adrian, ids_adrian, slices_paul, ids_paul, slices_sacha, ids_sacha, slices_h, ids_h)


    # slice analysis
    total_percentage, percentage, total_slices, total_slices_a, total_slices_p, total_slices_s, total_slices_h, no_patients = GetSliceOverlap4(PLIC_new_mask_paul,PLIC_new_id_paul,PLIC_new_mask_adrian,PLIC_new_id_adrian,PLIC_new_mask_sacha,PLIC_new_id_sacha, PLIC_new_mask_h, PLIC_new_id_h)
    slices_df = pd.DataFrame(percentage, columns = ["Slice IOU"], index = PLIC_new_id_adrian)
    slices_df["Total IOU"] = total_percentage
    slices_df.index.name = "Patient ID"
    print('total percentage: ',total_percentage)
    print("Total slices: ", total_slices, "Adrian slices: ", total_slices_a, "Paul slices: ", total_slices_p, "Sacha slices: ", total_slices_s, "H slices: ", total_slices_h)
    print("Number of patients: ", no_patients)

    # STAPLE analysis
    #staple_array, staple_ids, staple_slices = StapleGunPLIC( slices__a= new_slices_adrian, slices__b= new_slices_paul, slices__c= new_slices_sacha, slices__d= new_slices_h, p_masks_a= PLIC_new_mask_adrian, p_masks_b= PLIC_new_mask_paul, p_masks_c= PLIC_new_mask_sacha, p_masks_d= PLIC_new_mask_h, p_ids_a= PLIC_new_id_adrian, p_ids_b= PLIC_new_id_paul, p_ids_c= PLIC_new_id_sacha, p_ids_d= PLIC_new_id_h)
    #staple_array, staple_ids, staple_slices = StapleGun4(masks_a= PLIC_new_mask_adrian, masks_b= PLIC_new_mask_paul, masks_c= PLIC_new_mask_sacha, masks_d= PLIC_new_mask_h, ids_a= PLIC_new_id_adrian, ids_b = PLIC_new_id_paul, ids_c= PLIC_new_id_sacha, ids_d = PLIC_new_id_h)
    #staple_array, staple_ids, staple_slices = StapleGun(masks_a= new_mask_adrian, masks_b= new_mask_paul, masks_c= new_mask_h,  ids_a= new_id_adrian, ids_b = new_id_paul, ids_c= new_id_h)
    #staple_arr = SaveStapleData(staple_array, staple_ids, staple_slices)


    """

    dice_array_adrian = DiceCaster(staple_array, staple_ids, staple_slices, PLIC_new_mask_adrian, PLIC_new_id_adrian)
    adrian_total_avg, adrian_patient_avg = AverageDice(dice_array_adrian)
    adrian_df = pd.DataFrame(dice_array_adrian, columns = ["Dice 1", "Dice 2", "Dice 3" , "Dice 4" ] , index = staple_ids)
    #adrian_df.insert(0, "Patient_ID", new_id_adrian, True)
    adrian_df["Average Dice"] = adrian_patient_avg
    adrian_df.index.name = "Patient ID"

    dice_array_paul = DiceCaster(staple_array, staple_ids, staple_slices, PLIC_new_mask_paul, PLIC_new_id_paul)
    paul_total_avg, paul_patient_avg = AverageDice(dice_array_paul)
    paul_df = pd.DataFrame(dice_array_paul, columns = ["Dice 1", "Dice 2", "Dice 3", "Dice 4"], index = staple_ids)
    #paul_df.insert(0, "Patient_ID", new_id_paul, True)
    paul_df["Average Dice"] = paul_patient_avg
    paul_df.index.name = "Patient ID"

    dice_array_sacha = DiceCaster(staple_array, staple_ids, staple_slices, PLIC_new_mask_sacha, PLIC_new_id_sacha)
    print("staple slices: ", staple_slices)
    print("len staple: ", len(staple_slices))
    sacha_total_avg, sacha_patient_avg = AverageDice(dice_array_sacha)
    sacha_df = pd.DataFrame(dice_array_sacha, columns = ["Dice 1", "Dice 2", "Dice 3", "Dice 4"], index = staple_ids)
    #sacha_df.insert(0, "Patient_ID", new_id_sacha, True)
    sacha_df["Average Dice"] = sacha_patient_avg
    sacha_df.index.name = "Patient ID"
    
    dice_array_h = DiceCaster(staple_array, staple_ids, staple_slices, PLIC_new_mask_h, PLIC_new_id_h)
    h_total_avg, h_patient_avg = AverageDice(dice_array_h)
    h_df = pd.DataFrame(dice_array_h, columns = ["Dice 1", "Dice 2", "Dice 3", "Dice 4"], index = staple_ids)
    #sacha_df.insert(0, "Patient_ID", new_id_sacha, True)
    h_df["Average Dice"] = h_patient_avg
    h_df.index.name = "Patient ID"

    total_df = pd.DataFrame([adrian_total_avg, paul_total_avg, sacha_total_avg, h_total_avg], index =["Adrian STAPLE average Dice","Paul STAPLE average Dice", "Sacha STAPLE average Dice", "H STAPLE average dice"] )
   
    print("avg adrian: ", adrian_total_avg, "avg paul: ", paul_total_avg, "avg sacha: ", sacha_total_avg, "avg h: ", h_total_avg)
    """
    """
    with pd.ExcelWriter('/Users/olivia/Documents/PhD/MISTIE/mask_data/output_plic.xlsx') as writer:  
        adrian_df.to_excel(writer, sheet_name='Adrian')
        paul_df.to_excel(writer, sheet_name='Paul')
        sacha_df.to_excel(writer, sheet_name='Sacha')
        h_df.to_excel(writer, sheet_name='H')
        total_df.to_excel(writer, sheet_name='Total Dice')
        #slices_df.to_excel(writer,sheet_name ='Slices' )
    """
    """
    # plots 
    areas_s = GetAreaArray(PLIC_new_mask_sacha, PLIC_new_id_sacha, staple_ids, staple_slices)
    areas_a = GetAreaArray(PLIC_new_mask_adrian, PLIC_new_id_adrian, staple_ids, staple_slices)
    areas_p = GetAreaArray(PLIC_new_mask_paul, PLIC_new_id_paul, staple_ids, staple_slices)
    areas_h = GetAreaArray(PLIC_new_mask_h, PLIC_new_id_h, staple_ids, staple_slices)
    """
    """
    area_df = pd.DataFrame(areas,index = staple_ids)
    area_df.index.name = "Patient ID"
    with pd.ExcelWriter('/Users/olivia/Documents/PhD/MISTIE/mask_data/area.xlsx') as writer:  
        area_df.to_excel(writer, sheet_name='area')
    """
    """
    flattened_s = []
    area_flat_s =[]
    dice_flat_s =[]
    for i in range(0, len(dice_array_sacha)):
        for j in range(0, len(dice_array_sacha[i])):
            flat = (dice_array_sacha[i][j],areas_s[i][j] )
            dice = dice_array_sacha[i][j]
            area = areas_s[i][j]
            dice_flat_s.append(dice)
            area_flat_s.append(area)
            flattened_s.append(flat)

    flattened_a = []
    area_flat_a =[]
    dice_flat_a =[]
    for i in range(0, len(dice_array_adrian)):
        for j in range(0, len(dice_array_adrian[i])):
            flat = (dice_array_adrian[i][j],areas_a[i][j] )
            dice = dice_array_adrian[i][j]
            area = areas_a[i][j]
            dice_flat_a.append(dice)
            area_flat_a.append(area)
            flattened_a.append(flat)
    flattened_p = []
    area_flat_p =[]
    dice_flat_p =[]
    for i in range(0, len(dice_array_paul)):
        for j in range(0, len(dice_array_paul[i])):
            flat = (dice_array_paul[i][j],areas_p[i][j] )
            dice = dice_array_paul[i][j]
            area = areas_p[i][j]
            dice_flat_p.append(dice)
            area_flat_p.append(area)
            flattened_p.append(flat)
    flattened_h = []
    area_flat_h =[]
    dice_flat_h =[]
    for i in range(0, len(dice_array_h)):
        for j in range(0, len(dice_array_h[i])):
            flat = (dice_array_h[i][j],areas_h[i][j] )
            dice = dice_array_h[i][j]
            area = areas_h[i][j]
            dice_flat_h.append(dice)
            area_flat_h.append(area)
            flattened_h.append(flat)

    s = plt.scatter(*zip(*flattened_s))
    a = plt.scatter(*zip(*flattened_a))
    p = plt.scatter(*zip(*flattened_p))
    h = plt.scatter(*zip(*flattened_h))
    plt.plot(np.unique(flattened_s[0]), np.poly1d(np.polyfit(flattened_s[0], flattened_s[1], 1))(np.unique(flattened_s[0])))
    plt.xlim(0,1)
    plt.title("All")
    plt.xlabel("Dice score" )
    plt.ylabel("IC area in pixels")
    plt.legend(p.legend_elements(),
                        loc="lower left", title="Classes")

    #plt.show()
    dices = dice_flat_a + dice_flat_p + dice_flat_s +dice_flat_h
    #dicess.append(flattened_s)
    print((sum(dice_flat_a)/len(dice_flat_a)), (sum(dice_flat_p)/len(dice_flat_p)), (sum(dice_flat_s)/len(dice_flat_s)) ,(sum(dice_flat_h)/len(dice_flat_h)))
    areas = area_flat_a + area_flat_p + area_flat_s + area_flat_h
    no_names_a = len(dice_flat_a)
    no_names_p = len(dice_flat_p)
    no_names_s = len(dice_flat_s)
    no_names_h = len(dice_flat_h)
    print(len(dice_flat_a))
    print("no names: ", no_names_a)
    adrians =  ["Adrian"]* no_names_a
    pauls = ["Paul"]*no_names_p
    sachas = ["Sacha"]*no_names_s
    hs = ["H"]*no_names_h
    observers = adrians + pauls + sachas +hs
    print(observers)
    

    total_dataframe = pd.DataFrame(dices, columns= ["Dice"])
    total_dataframe["Observers"] = observers
    import seaborn as sns
    sns.set_theme(style ="ticks")
    
    g = sns.catplot(y = 'Dice', x="Observers", data= total_dataframe,  kind = "swarm")
   
    #sns.catplot(y = "Dice", data = total_dataframe, hue = "Observers", kind = "swarm")
    plt.show()


    '''
    total_dataframe["Areas"] = areas
    

   

    #plt.boxplot(flattened_a)
    data = [adrian_patient_avg, sacha_patient_avg, paul_patient_avg, h_patient_avg]

    df = pd.DataFrame(adrian_patient_avg, columns = ['Adrian'])
    df['Paul'] = paul_patient_avg
    df['Sacha'] = sacha_patient_avg
    df['H'] = h_patient_avg

    boxplot = df.boxplot() 
    plt.show()
    fig = boxplot.get_figure()
    fig.savefig('/Users/olivia/Documents/PhD/MISTIE/mask_data/boxplot.png')
   
    # Plot sepal width as a function of sepal_length across days
    g = sns.lmplot(
        data = total_dataframe,
        x="Dice", y="Areas", hue = "Observers"
    )
     
    # Use more informative axis labels than are provided by default
    g.set_axis_labels("Dice score", "Area (pixels)")
    plt.show()
    sns.set_theme(style="ticks")
    sns.pairplot(total_dataframe, hue="Observers")
    plt.show()
    sns.catplot(total_dataframe, hue= "Observers")
    '''
    """
if __name__ == '__main__':
 main()