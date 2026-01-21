#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import argparse
import nibabel as nib
import numpy as np
from dipy.tracking._utils import _mapping_to_voxel, _to_voxel_coordinates
import subprocess
import os
import ants
from tqdm import tqdm
import requests
import shutil
import glob
from sklearn.cluster import DBSCAN
import json

def buildArgsParser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)


    p.add_argument('ID',
                   help='Subject ID')
    p.add_argument('input',
                   help='Input folder with DICOMs)')
    p.add_argument('output_dir',
                   help='Specify output directory (derivatives)')
    p.add_argument('side',
                   help='Specify affected hemisphere: 1=Right, 2=Left')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()


    #datalist
    MNI_HCP = 'mni_icbm152_t1_tal_nlin_asym_09c.nii'#'MNI152_T1_1mm.nii.gz'#
    MNI_HCP_brain = 'mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz'  # 'MNI152_T1_1mm.nii.gz'#
    MNI_brainmask = 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'#'MNI152_T1_1mm.nii.gz'#
    reference = "HCPA422-T1w-500um-norm.nii.gz"
    LH_Brain = 'LH_hcp.nii.gz'#'LH_brain.nii.gz'#
    RH_Brain = 'RH_hcp.nii.gz'#'RH_brain.nii.gz'#

    #out_NT_disc = os.path.join(args.output_dir, args.ID + "_NT_Diconnect_" + args.NTmaps + ".csv")


    ### Extract nifti
    output_direct = os.path.join(args.output_dir, args.ID)

    if not os.path.exists(output_direct):
        os.makedirs(output_direct)



    ###### dcm2nii
    cbf = os.path.join(output_direct, "sub-" + args.ID + "_cbf.nii.gz")
    tmax = os.path.join(output_direct, "sub-" + args.ID + "_tmax.nii.gz")
    cbv = os.path.join(output_direct, "sub-" + args.ID + "_cbv.nii.gz")
    raw = os.path.join(output_direct, "sub-" + args.ID + "_raw.nii.gz")
    if os.path.isfile(cbf) == True:
        print("DCM2NII already performed")
    else:
        #dcm2nii_cmd = "dcm2niix -b y -z y -f " + args.ID + "_%d -o " + args.input + " " + args.input
        #subprocess.call(dcm2nii_cmd, shell=True)

        print(glob.glob(os.path.join(args.input, "*MIP*nii.gz")))
        print(glob.glob(os.path.join(args.input, "*TMAX*nii.gz")))

        nii_file = glob.glob(os.path.join(args.input, "*TMAX*nii.gz"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, tmax)
        nii_file = glob.glob(os.path.join(args.input, "*TMAX*json"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, tmax[:-7]+".json")
        nii_file = glob.glob(os.path.join(args.input, "*CBF*nii.gz"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, cbf)
        nii_file = glob.glob(os.path.join(args.input, "*CBF*json"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, cbf[:-7] + ".json")
        nii_file = glob.glob(os.path.join(args.input, "*CBV*nii.gz"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, cbv)
        nii_file = glob.glob(os.path.join(args.input, "*CBV*json"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, cbv[:-7] + ".json")
        nii_file = glob.glob(os.path.join(args.input, "*MIP_#*nii.gz"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        print(nii_file)
        shutil.copy(nii_file, raw)
        nii_file = glob.glob(os.path.join(args.input, "*MIP_#*json"))
        nii_file = [f for f in nii_file if "RGB" not in os.path.basename(f) and "RAPID" not in os.path.basename(f)][0]
        shutil.copy(nii_file, raw[:-7] + ".json")

    raw_preproc = os.path.join(output_direct, "sub-" + args.ID + "_raw_preproc.nii.gz")
    if os.path.isfile(raw_preproc) == True:
        print("Preproc raw already performed")
    else:
        ########## preprocess raw ####################
        print("Processing raw image")
        img_raw = nib.load(raw)
        img_data = img_raw.get_fdata()

        a = np.where((img_data > -1000) & (img_data < -100))
        b = np.where((img_data > -99) & (img_data < 100))
        b_data = img_data[b]
        b_interp = np.interp(b_data, (b_data.min(), b_data.max()), (911, 3100))
        c = np.where(img_data > 100)
        img_data[a] += 1000
        img_data[b] = b_interp
        img_data[c] += 3000

        img_new = nib.Nifti1Image(img_data, img_raw.affine, img_raw.header)

        nib.save(img_new, raw_preproc)

    ###############Registration to Standard
    ### brain mask

    if os.path.isfile(os.path.join(output_direct, "sub-" + args.ID + "_cbf_inMNI.nii.gz")) == True:
        print("Coregistration already performed")
    else:
        cbf_info = nib.load(cbf)
        cbf_mask = cbf_info.get_fdata()
        cbf_mask[cbf_mask > 100] = np.nan
        cbf_mask[cbf_mask < 0] = np.nan
        mask = nib.Nifti1Image(cbf_mask, cbf_info.affine, cbf_info.header)
        CT_mask = os.path.join(output_direct, "sub-" + args.ID + "_CTP_mask.nii.gz")
        nib.save(mask, CT_mask)

        ###Coregist
        print("Coregistration to HCP standard brain")

        MNI_info = nib.load(MNI_HCP)
        fi = ants.image_read(MNI_HCP)
        mi = ants.image_read(raw_preproc)
        mask = ants.image_read(CT_mask)

        mat = os.path.join(output_direct, "sub-" + args.ID + "_CTP_affine_HCP.mat")
        reg_out = os.path.join(output_direct, "sub-" + args.ID + "_CTP_inMNI.nii.gz")
        warp_out = os.path.join(output_direct, "sub-" + args.ID + "_CTP_inMNI_Warp.nii.gz")

        shutil.move(ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', moving_mask=mask)['fwdtransforms'][1], mat)
        tx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', moving_mask=mask)
        warped = tx['warpedmovout']
        ants.image_write(warped, reg_out)
        forwardtrans = tx['fwdtransforms']
        #invtrans = tx['invtransforms']
        # invWarp = tx['warpedfixout']
        shutil.move(forwardtrans[0], warp_out)

        ### apply transform
        listtransf = [str(warp_out), str(mat)]

        for i in [cbf, tmax]:
            reg_out = i[:-7]+"_inMNI.nii.gz"
            movmapimg = i
            movmap = ants.image_read(movmapimg)
            mywarpedimage = ants.apply_transforms(fixed=fi, moving=movmap,
                                                  transformlist=listtransf, interpolator='linear')
            ants.image_write(mywarpedimage, reg_out)

        '''
        mov_data = nib.load(reg_out).get_fdata()
        mov_data[mov_data <= 0] = np.nan
        # test_r = conform(test, out_shape=(182, 218, 182), voxel_size=(1.0, 1.0, 1.0), order=3, orientation='LAS')
        movmapimg = nib.Nifti1Image(mov_data, MNI_info.affine, MNI_info.header)
        nib.save(movmapimg, os.path.join(registf, id_img + '_CTP_' + i + '_inMNI_HCP.nii.gz'))
        '''

    ###### Create Lesion masks



    thrcluster = 80

    strokemask = os.path.join(output_direct, "masks")
    if not os.path.exists(strokemask):
        os.makedirs(strokemask)

    CTMNI = MNI_HCP
    MNI_info = nib.load(CTMNI)
    MNI_brain = MNI_brainmask
    brainmask = nib.load(MNI_brain).get_fdata()


    AH = [RH_Brain, LH_Brain]
    UH = [LH_Brain, RH_Brain]
    AH_brain = nib.load(AH[int(args.side) - 1]).get_fdata()
    UH_brain = nib.load(UH[int(args.side )- 1]).get_fdata()

    #### CBF <= 30% mask Yu 2015

    cbf_mni = os.path.join(output_direct, "sub-" + args.ID + "_cbf_inMNI.nii.gz")
    cbf_data = nib.load(cbf_mni).get_fdata()
    #cbf_data[cbf_data == 0] = np.nan
    #cbf_data[brainmask == 0] = np.nan
    overlay =(UH_brain != 0) & (brainmask != 0) & (cbf_data > 0)
    meancbf = np.nanmean(cbf_data[overlay])
    #meancbf = np.nanmean(cbf_data[UH_brain == 1])
    cbf_data[AH_brain == 0] = np.nan
    print(meancbf)
    cbf_data[cbf_data > (0.3 * meancbf)] = np.nan
    cbf_data = np.where(cbf_data > 0, 1, np.nan)
    print(np.count_nonzero(cbf_data))
    # cbf_data[cbf_data > 0] == 1
    new_image = np.full(np.shape(cbf_data), np.nan)
    for zachse in np.arange(0, len(cbf_data[0, 0, :])):
        slice = cbf_data[:, :, zachse]
        indices = np.argwhere(slice[:, :] == 1)
        if indices.size > 0:
            db = DBSCAN(eps=2, min_samples=2).fit(indices)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            validclusters = []
            for la in np.arange(0, n_clusters_):
                if np.count_nonzero(labels == la) > thrcluster:
                    # print("Nr of Voxels for valid Cluster at z = "+str(zachse))
                    # print(np.count_nonzero(labels== la))
                    validclusters.append(la)
                    # clustersizes.append(np.count_nonzero(labels== la))
            test = np.isin(labels, validclusters).astype(int)
            z = np.full((len(labels), 1), zachse)
            new_image[indices[:, 0], indices[:, 1], z] = test
    new_image = np.where(new_image > 0, 1, np.nan)

    img_cbf = nib.Nifti1Image(new_image, MNI_info.affine, MNI_info.header)
    nib.save(img_cbf, os.path.join(strokemask, "sub-" + args.ID + "_cbf30_lesion.nii.gz"))

    tmax_mni = os.path.join(output_direct, "sub-" + args.ID + "_tmax_inMNI.nii.gz")
    tmax_data = nib.load(tmax_mni).get_fdata()
    tmax_data[brainmask == 0] = np.nan
    tmax_data[AH_brain == 0] = np.nan
    tmax_data[tmax_data <= 6] = np.nan
    tmax_data = np.where(tmax_data > 0, 1, np.nan)
    new_image = np.full(np.shape(tmax_data), np.nan)
    for zachse in np.arange(0, len(tmax_data[0, 0, :])):
        slice = tmax_data[:, :, zachse]
        indices = np.argwhere(slice[:, :] == 1)
        if indices.size > 0:
            db = DBSCAN(eps=2, min_samples=2).fit(indices)
            labels = db.labels_
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            validclusters = []
            for la in np.arange(0, n_clusters_):
                if np.count_nonzero(labels == la) > thrcluster:
                    # print("Nr of Voxels for valid Cluster at z = "+str(zachse))
                    # print(np.count_nonzero(labels== la))
                    validclusters.append(la)
                    # clustersizes.append(np.count_nonzero(labels== la))
            test = np.isin(labels, validclusters).astype(int)
            z = np.full((len(labels), 1), zachse)
            new_image[indices[:, 0], indices[:, 1], z] = test
    new_image = np.where(new_image > 0, 1, np.nan)

    img_tmax = nib.Nifti1Image(new_image, MNI_info.affine, MNI_info.header)
    nib.save(img_tmax,  os.path.join(strokemask, "sub-" + args.ID + "_tmax6_lesion.nii.gz"))

    ##### Define penumbra
    image_tmax = img_tmax.get_fdata()
    image_cbf = img_cbf.get_fdata()
    penumbra = np.zeros(np.shape(image_tmax))  # = image_tmax - image_cbf
    penumbra[image_tmax > 0] = 1
    penumbra[image_cbf > 0] = 0
    penumbra[penumbra < 0] = 0
    image3 = nib.Nifti1Image(penumbra, MNI_info.affine, MNI_info.header)
    nib.save(image3, os.path.join(strokemask, "sub-" + args.ID + "_penumbra.nii.gz"))



    def define_streamlines(streamlines, lesion, reference):  # , NT_weights_SUM):#, weights):
        metric_tractrogram = []
        les = lesion.get_fdata()
        les[:,:,0:153] = 0 #set all values below the chiasma opticum to zero (artefacts, infratentoriel outside media territorium)
        affine = reference.affine
        lin_T, offset = _mapping_to_voxel(affine)

        for s in tqdm(range(2000000), desc="Evaluate streamlines"):
            streamline = streamlines[s]
            ### location
            x_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 0]
            y_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 1]
            z_ind_2 = _to_voxel_coordinates(streamline[:], lin_T, offset)[:, 2]

            if np.sum(les[x_ind_2, y_ind_2, z_ind_2]) > 0:
                metric_tractrogram.append(1)
                # metric_tractogram_preserved.append(0)
            else:
                metric_tractrogram.append(0)
            #    metric_tractogram_preserved.append(a[s])

        return (metric_tractrogram)


    tck_file = "HCP422_2_million.tck"
    if os.path.isfile(tck_file):
        print("Tactogram exists")
    else:
        print("Downloading Tractogram...........")
        osf_url = "https://osf.io/download/nduwc/"
        response = requests.get(osf_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(tck_file, "wb") as file, tqdm(desc=tck_file, total=total_size, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
        print("Download complete!")

    ### Create Warpfield for MNI to HCPA transformation
    if os.path.isfile('MNI_to_HCPA_Warp.nii.gz'):
        print("Coregistration MNI to HCPA done")
    else:
        print("Coregistration MNI to HCPA .....")
        mi = ants.image_read(MNI_HCP_brain)
        fi = ants.image_read(reference)
        tx = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', random_seed=1, singleprecision=False)
        forwardtrans = tx['fwdtransforms']
        forwardtrans = tx['fwdtransforms']
        shutil.copyfile(forwardtrans[1], "MNI_to_HCPA.mat")
        shutil.copyfile(forwardtrans[0], "MNI_to_HCPA_Warp.nii.gz")
        print("Coregistration done!")

    disc = os.path.join(output_direct, "disc")
    if not os.path.exists(disc):
        os.makedirs(disc)

    if os.path.isfile(os.path.join(disc, "cbf_Disc_SL.txt")) == True:
        print("disc sl already calculated")
    else:

        print("Loading streamlines ##########################################")
        tractogram = nib.streamlines.load(tck_file)
        streamlines = tractogram.streamlines
        header_sl = tractogram.header



        ### Bring Input Lesion (MNI) in HCPA MNI
        maps = ["cbf", "tmax", "penumbra"]
        for ix, i in enumerate([os.path.join(strokemask, "sub-" + args.ID + "_cbf30_lesion.nii.gz"), os.path.join(strokemask, "sub-" + args.ID + "_tmax6_lesion.nii.gz"), os.path.join(strokemask, "sub-" + args.ID + "_penumbra.nii.gz")]):
            print("Calculating "+maps[ix]+" ...........")
            out_weights_tractogram_disc = os.path.join(disc, maps[ix] + "_Disc_SL.txt")

            standard = "HCPA422-T1w-500um-norm.nii.gz"
            listtransf = ['MNI_to_HCPA_Warp.nii.gz', "MNI_to_HCPA.mat"]
            fi = ants.image_read(standard)
            movmap = ants.image_read(i)
            mywarpedimage = ants.apply_transforms(fixed=fi, moving=movmap,
                                                  transformlist=listtransf, interpolator='multiLabel')

            output = mywarpedimage.numpy()
            ref = nib.load(standard)
            lesion = nib.Nifti1Image(output, ref.affine, ref.header)
            nib.save(lesion, "tmp_les.nii.gz")

            weights_tractogram = define_streamlines(streamlines, lesion, nib.load(standard))
            np.savetxt(out_weights_tractogram_disc, weights_tractogram)




if __name__ == "__main__":
    main()

# %%