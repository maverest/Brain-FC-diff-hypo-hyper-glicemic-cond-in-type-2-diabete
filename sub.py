from fsl.wrappers import (
    fslreorient2std, mcflirt, bet, fast, epi_reg, flirt, fnirt, applywarp, fsl_anat, fslmaths
)

from pathlib import Path
import subprocess as sp
import copy
import shutil
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image as nimg, masking
from nilearn.plotting import plot_design_matrix
import time
import contextlib
import sys
from contextlib import contextmanager
from tqdm import tqdm


@contextmanager
def suppress_stdout_stderr():
    """Temporarily redirect stdout and stderr to os.devnull."""
    devnull = open(os.devnull, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

# Define paths and constants
DATA = Path(__file__).parent / "images"
DERIVATE = Path(__file__).parent / "derivates"
MNI_HEAD_2mm = Path(os.environ.get("FSLDIR", "/usr/local/fsl") + "/data/standard/MNI152_T1_2mm.nii.gz")
MNI_BRAIN_2mm = Path(os.environ.get("FSLDIR", "/usr/local/fsl") + "/data/standard/MNI152_T1_2mm_brain.nii.gz")

class AnatT1:
    def __init__(self, sub_id : str, condition: str ):
        self.sub_id = sub_id
        self.condition = condition
        self.path_raw = DATA / sub_id / "anat" / condition
        self.path_der = DERIVATE / sub_id / "anat"/ condition 
        
        # Path that will be created during processing
        
        # reorient
        self.raw_img = list(self.path_raw.glob("*.nii.gz"))[0]
        self.t1_img_or = self.path_der / self.raw_img.stem.replace("nii", "nii.gz")
        
        # BET
        self.t1_img_or_brain = Path(str(self.t1_img_or).replace(".nii.gz", "_brain.nii.gz"))
        self.t1_img_or_brain_mask = Path(str(self.t1_img_or).replace(".nii.gz", "_brain_mask.nii.gz"))
        
        # FAST
        fast_path = self.path_der / "fast"
        fast_path.mkdir(exist_ok=True)
        self.path_t1_brain_fast = fast_path / str(self.t1_img_or_brain.name.replace(".nii.gz", "_fast"))
        self.t1_wm = fast_path / str(self.t1_img_or_brain.name.replace(".nii.gz", "_fast_pve_2.nii.gz"))
        self.t1_csf = fast_path / str(self.t1_img_or_brain.name.replace(".nii.gz", "_fast_pve_0.nii.gz"))
        self.t1_gm = fast_path / str(self.t1_img_or_brain.name.replace(".nii.gz", "_fast_pve_1.nii.gz"))

        # T1 to MNI
        t1_to_mni_folder = self.path_der / "t1_to_mni"
        aff_trans_folder = t1_to_mni_folder / "linear"    
        t1_to_mni_folder.mkdir(exist_ok=True)    
        aff_trans_folder.mkdir(exist_ok=True)
        self.t1_to_mni_aff_mat = aff_trans_folder / "t1_to_mni_aff.mat"
        self.t1_img_or_brain_mni_aff = self.path_der / str(self.t1_img_or_brain.name.replace(".nii.gz", "_mni_aff.nii.gz"))
        
        nnlin_trans_folder = t1_to_mni_folder / "nonlinear"
        nnlin_trans_folder.mkdir(exist_ok=True)
        self.t1_to_mni_nnlin_warp = nnlin_trans_folder / "t1_to_mni_nnlin_warp.nii.gz"
        self.t1_to_mni_nnlin_coef = nnlin_trans_folder / "t1_to_mni_nnlin_field.nii.gz"
        self.t1_img_or_head_mni_nnlin = self.path_der / str(self.t1_img_or_brain.name.replace("brain.nii.gz", "head_mni_nnlin.nii.gz"))
        
        regressor_dir = self.path_der / "regressors"
        regressor_dir.mkdir(exist_ok=True)
        self.wm_regressor = regressor_dir / "WM_T1_regressor.nii.gz"
        self.csf_regressor = regressor_dir / "CSF_T1_regressor.nii.gz"
        
        self.t1_gm_mni = self.path_der / str(self.t1_gm.name.replace("fast_pve_1.nii.gz", "gm_to_mni.nii.gz"))
        self.t1_gm_mask_thr = self.path_der / str(self.t1_gm.name.replace("fast_pve_1.nii.gz", "gm_to_mni_thr.nii.gz"))
        

    def reorient(self):
        fslreorient2std(self.raw_img, self.t1_img_or) # Reorient to standard

    def brain_extraction(self, R = True, frac = 0.35, mask  =True, display = False):
        bet(self.t1_img_or, self.t1_img_or_brain, R = R, f = frac, mask = mask) #B = True) 
        if display:
            sp.Popen(["fsleyes",
                        self.t1_img_or_brain_mask,
                        self.t1_img_or,
                        self.t1_img_or_brain, "-cm", "green","-a", "40",
                      ])
        
    def tissue_segmentation(self, segments = 3):
        fast(imgs =self.t1_img_or_brain,out = self.path_t1_brain_fast,B = True, t = 1, segments = segments)


    def t1_to_mni(self):
        flirt(src=self.t1_img_or_brain, ref=MNI_BRAIN_2mm, omat=self.t1_to_mni_aff_mat,out = self.t1_img_or_brain_mni_aff ,dof=12)
        fnirt(src=self.t1_img_or, aff=self.t1_to_mni_aff_mat, fout=self.t1_to_mni_nnlin_warp, config="T1_2_MNI152_2mm.cnf", iout=self.t1_img_or_head_mni_nnlin, ref=MNI_HEAD_2mm, cout=self.t1_to_mni_nnlin_coef)

    def make_regressors_mask(self, th =0.9):
        fslmaths(Path(self.t1_wm)).thr(th).bin().ero().ero().run(self.wm_regressor)
        fslmaths(Path(self.t1_csf)).thr(th).bin().ero().ero().run(self.csf_regressor)
        
    def gm_mask_to_mni(self):
        applywarp(
            src=self.t1_gm,
            ref =MNI_HEAD_2mm,
            warp = self.t1_to_mni_nnlin_warp,
            out  = self.t1_gm_mni,
            interp = "spline")
        
    def threshold_gm_mask(self, th=0.7):
        gm_img = nib.load(self.t1_gm_mni)
        gm_data = gm_img.get_fdata()
        gm_data_thr = (gm_data >= th).astype(np.uint8)
        gm_mask_img = nib.Nifti1Image(gm_data_thr, gm_img.affine, gm_img.header)
        gm_mask_img.set_data_dtype(np.uint8)
        out_path = self.t1_gm_mask_thr
        nib.save(gm_mask_img, out_path)
            
    def run_anat_pipeline(self):
        print(f"\n=== Running full anatomical pipeline for {self.sub_id} [{self.condition}] ===")
        t0 = time.perf_counter()

        step_start = time.perf_counter()
        print(f"  * [1/4] Reorient...", end="", flush=True)
        self.reorient()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [2/4] Brain extraction...", end="", flush=True)
        self.brain_extraction()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [3/4] Tissue segmentation...", end="", flush=True)
        self.tissue_segmentation()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [4/4] T1 to MNI...", end="", flush=True)
        self.t1_to_mni()
        self.make_regressors_mask()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        total = time.perf_counter() - t0
        print(f"=== Done: {self.sub_id} [{self.condition}] ({total:.1f}s total) ===\n")

        
    def visualize(self):

        brain  = self.path_der / f"{self.sub_id}_{self.condition.capitalize()}_T1w_brain.nii.gz"
        wm = self.path_der / "fast" / str(self.t1_img_or.name.replace(".nii.gz", "_brain_fast_pve_2.nii.gz"))
        csf = self.path_der / "fast" / str(self.t1_img_or.name.replace(".nii.gz", "_brain_fast_pve_0.nii.gz"))
        gm = self.path_der / "fast" / str(self.t1_img_or.name.replace(".nii.gz", "_brain_fast_pve_1.nii.gz"))
        lin_brain = self.path_der / f"{self.sub_id}_{self.condition.capitalize()}_T1w_brain_mni_aff.nii.gz"
        nn_lin_head = self.path_der / f"{self.sub_id}_{self.condition.capitalize()}_T1w_head_mni_nnlin.nii.gz"

        sp.Popen(["fsleyes",
                    self.t1_img_or,
                    brain,
                    wm,
                    csf,
                    gm,
                    lin_brain,
                    nn_lin_head,
                    MNI_HEAD_2mm
                  ])


class funcFMRI:
    def __init__(self, sub_id : str, condition: str , run : str,): 
        self.anat_data = AnatT1(sub_id, condition)
        self.run = str(run).capitalize()
        self.sub_id = sub_id
        self.condition = condition
        self.path_raw = DATA / sub_id / "func" / condition
        self.path_der = DERIVATE / sub_id / "func"/ condition 
        
        self.run_dir = self.path_der / f"run{self.run}"
        self.run_dir.mkdir(exist_ok=True)
        img = self.path_raw.glob("*.nii.gz")
        for file in img : 
            if f"run{self.run}" in str(file):
                self.raw_img = file
        self.TR = nib.load(self.raw_img).header.get_zooms()[3]
        self.func_img_or = self.run_dir / self.raw_img.stem.replace("nii", "nii.gz")
        
        
        # Path that will be created during processing
        
        # reorient
        self.func_img_or = self.run_dir / self.raw_img.stem.replace("nii", "nii.gz")
        
        # motion correction
        self.mc_dir = self.run_dir / "mc"
        self.mc_dir.mkdir(exist_ok=True)
        self.func_img_or_mc = self.mc_dir / str(self.func_img_or.name.replace(".nii.gz", "_mc.nii.gz")) # 4D images MC
        self.func_img_or_mc_mean = self.mc_dir / str(self.func_img_or.name.replace(".nii.gz", "_mc_mean_reg.nii.gz")) # Mean of 4D images MC
        self.func_img_or_mc_par = self.mc_dir / str(self.func_img_or.name.replace(".nii.gz", "_mc.par")) # 2D text file with 6 motion parameters (EPI)
        #! ajouter les mat ou par ? 

        # epi to t1
        self.epireg_dir = self.run_dir / "epireg"
        self.epireg_dir.mkdir(exist_ok=True)
        self.func_img_or_mc_t1 = self.epireg_dir / str(self.func_img_or_mc.name.replace(".nii.gz", "_epireg.nii.gz"))
        self.epi_to_t1_mat = self.epireg_dir / str(self.func_img_or_mc.name.replace(".nii.gz", "_epireg.mat"))

        self.t1_to_epi_dir = self.run_dir / "t1_to_epi"
        self.t1_to_epi_dir.mkdir(exist_ok=True)

        self.t1_to_epi_mat = self.t1_to_epi_dir / "t1_to_epi.mat"
        
        
        self.t1_to_epi_mask = self.t1_to_epi_dir / self.anat_data.t1_img_or_brain.name.replace(".nii.gz", f"_t1_to_epi_mask_{self.run}.nii.gz")
        self.func_img_or_mc_bet = Path(self.run_dir / self.func_img_or_mc.name.replace(".nii.gz", f"_bet.nii.gz"))
        
        regressor_dir = self.run_dir / "regressors"
        regressor_dir.mkdir(exist_ok=True)
        self.regressor_dir = regressor_dir
        self.wm_regressor = regressor_dir / f"WM_epi_regressor_{self.run}.nii.gz"
        self.csf_regressor = regressor_dir / f"CSF_epi_regressor_{self.run}.nii.gz"
        
        self.glm_output_dir = self.run_dir / "glm_output"
        self.glm_output_dir.mkdir(exist_ok=True)
        self.func_img_or_mc_bet_mean = self.glm_output_dir / str(self.func_img_or_mc_bet.name.replace(".nii.gz", "_mean.nii.gz"))
        self.func_img_or_mc_bet_clean_withGSR = self.glm_output_dir / self.func_img_or_mc_bet.name.replace(".nii.gz", f"_clean_withgsr.nii.gz")
        self.func_img_or_mc_bet_clean_withoutGSR = self.glm_output_dir / self.func_img_or_mc_bet.name.replace(".nii.gz", f"_clean_nogsr.nii.gz")
        
        self.t1_to_epi_to_mni_mask = self.run_dir / Path(self.t1_to_epi_mask).name.replace(f"_t1_to_epi_mask_{self.run}.nii.gz", f"_t1_to_epi_to_mni_mask_{self.run}.nii.gz")
        self.func_img_or_mc_bet_mean_mni = self.run_dir / str(self.func_img_or_mc_bet_mean.name.replace(".nii.gz", "_mni.nii.gz"))

        self.func_img_or_mc_bet_clean_withGSR_mni = self.run_dir / str(Path(self.func_img_or_mc_bet_clean_withGSR).name.replace(".nii.gz", "_mni.nii.gz"))
        self.func_img_or_mc_bet_clean_withoutGSR_mni = self.run_dir / str(Path(self.func_img_or_mc_bet_clean_withoutGSR).name.replace(".nii.gz", "_mni.nii.gz"))
        self.func_img_or_mc_bet_clean_withGSR_mni_smooth = self.run_dir / str(Path(self.func_img_or_mc_bet_clean_withGSR_mni).name.replace(".nii.gz", "_smooth.nii.gz"))
        self.func_img_or_mc_bet_clean_withoutGSR_mni_smooth = self.run_dir / str(Path(self.func_img_or_mc_bet_clean_withoutGSR_mni).name.replace(".nii.gz", "_smooth.nii.gz"))

    @staticmethod
    def make_confounds(include_gsr, wm_ts, csf_ts, gs_ts, mot6, lin, quad):
        cols = [
            wm_ts.reshape(-1,1),
            csf_ts.reshape(-1,1),
            mot6,
            lin,
            quad
        ]
        labels = ["WM", "CSF"] + [f"mot{i+1}" for i in range(6)] + ["lin", "quad"]
        if include_gsr:
            cols.insert(2, gs_ts.reshape(-1,1))
            labels.insert(2, "GSR")
        X = np.column_stack(cols).astype(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-8)
        df = pd.DataFrame(X, columns=labels)
        return df 
    


    def reorient(self):
        fslreorient2std(self.raw_img, self.func_img_or) 
 

    def motion_correction(self, dof=6):
        mc_out = self.mc_dir / self.func_img_or.name.replace(".nii.gz", "_mc")
        # with open(os.devnull, 'w') as devnull, \
        #     contextlib.redirect_stdout(devnull), \
        #     contextlib.redirect_stderr(devnull):
        mcflirt(
            infile=self.func_img_or,
            o=mc_out,
            mats=True,
            plots=True,
            report=True,
            dof=dof,
            meanvol=True,
        )



    def func_to_t1(self):
        out = self.epireg_dir / str(self.func_img_or_mc.name.replace(".nii.gz", "_epireg"))
        epi_reg(
            epi=self.func_img_or_mc_mean,
            t1=self.anat_data.t1_img_or,
            t1brain=self.anat_data.t1_img_or_brain,
            out=out,
        )
        
        
    def t1_mask_to_epi(self):
        mat_t1_to_epi = np.linalg.inv(np.loadtxt(self.epi_to_t1_mat)) 
        np.savetxt(self.t1_to_epi_mat, mat_t1_to_epi, fmt="%.6f")
        mask_t1 = self.anat_data.t1_img_or_brain_mask
        out = self.t1_to_epi_mask
        ref  = self.func_img_or_mc_mean

        flirt(  src=mask_t1,
                ref=ref,
                applyxfm=True, init=self.t1_to_epi_mat,
                interp="nearestneighbour",
                out=out)
        
        fslmaths(self.func_img_or_mc).mas(self.t1_to_epi_mask).run(self.func_img_or_mc_bet)
        
        
    def nuissance_regression(self, plot_design = True):

    # register the regressors from T1 to EPI space
        flirt( 
            src=self.anat_data.wm_regressor,  
            ref=self.func_img_or_mc_mean,
            applyxfm=True, init=self.t1_to_epi_mat,
            interp="nearestneighbour",
            out=self.wm_regressor)

        flirt(
            src=self.anat_data.csf_regressor, 
            ref=self.func_img_or_mc_mean,
            applyxfm=True, 
            init=self.t1_to_epi_mat,
            interp="nearestneighbour", 
            out=self.csf_regressor)

        
        fmri_img   = self.func_img_or_mc_bet  # motion-corrected + BET 4D EPI
        brainmask  = self.t1_to_epi_mask  # 3D binary mask in EPI space
        wm_mask    = self.wm_regressor  # 3D binary WM mask (EPI)
        csf_mask   = self.csf_regressor  # 3D binary CSF mask (EPI)
        mot6_file  = self.func_img_or_mc_par  # 2D text file with 6 motion parameters (EPI)
        TR         = self.TR  # Repetition time (s)
        drop       = 5    # drop initial volumes
                    
        # Band-pass 
        high_pass  = 0.005
        low_pass   = 0.1

        raw_img = nib.load(fmri_img)
        T_full  = raw_img.shape[-1] # numb of volumes
        func_img = raw_img.slicer[:, :, :, drop:]  # drop initial volumes 
        T = func_img.shape[-1]

        # Mean WM / CSF / Global signals (3 regressors)
        wm_ts  = masking.apply_mask(func_img, wm_mask).mean(axis=1)     # (T,)
        csf_ts = masking.apply_mask(func_img, csf_mask).mean(axis=1)    # (T,)
        gs_ts  = masking.apply_mask(func_img, brainmask).mean(axis=1)   # (T,)

        mot6 = np.loadtxt(mot6_file)[:T_full, :]
        if drop > 0:
            mot6 = mot6[drop:, :]
        assert mot6.shape == (T, 6), f"Motion shape mismatch: {mot6.shape} vs expected {(T,6)}" # check shape

        # Linear & quadratic trends 
        t = np.linspace(-1.0, 1.0, T)[:, None]
        lin  = t
        quad = t**2

        conf_with_gsr    = self.make_confounds(include_gsr=True, wm_ts=wm_ts, csf_ts=csf_ts, gs_ts=gs_ts, mot6=mot6, lin=lin, quad=quad)
        conf_without_gsr = self.make_confounds(include_gsr=False, wm_ts=wm_ts, csf_ts=csf_ts, gs_ts=gs_ts, mot6=mot6, lin=lin, quad=quad)

        # Optional: save for provenance
        conf_with_gsr.to_csv(self.regressor_dir / "confounds_withGSR.tsv", sep="\t", index=False)
        conf_without_gsr.to_csv(self.regressor_dir / "confounds_noGSR.tsv",  sep="\t", index=False)


        clean_withGSR = nimg.clean_img(
            func_img,
            confounds=conf_with_gsr.values,
            detrend=False,
            standardize=True, # z-score time vise
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=TR,
            mask_img=brainmask,
            clean__standardize_confounds= False #done manualy before so I can check the design matrix easily
        )

        clean_noGSR = nimg.clean_img(
            func_img,
            confounds=conf_without_gsr.values,
            detrend=False,
            standardize=True, 
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=TR,
            mask_img=brainmask,
            clean__standardize_confounds= False #done manualy before so I can check the design matrix easily
        )

        # Save cleaned images (the residuals of the regression)
        clean_withGSR.to_filename(self.func_img_or_mc_bet_clean_withGSR)
        clean_noGSR.to_filename(self.func_img_or_mc_bet_clean_withoutGSR)

        # Also save a mean EPI (with the bet) on avait pas celui la avant
        nimg.mean_img(func_img).to_filename(self.func_img_or_mc_bet_mean)

        if plot_design:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            plot_design_matrix(conf_with_gsr, rescale=True, ax=axes[0])
            axes[0].set_title("Design matrix (with GSR)")
            plot_design_matrix(conf_without_gsr, rescale=True, ax=axes[1])
            axes[1].set_title("Design matrix (no GSR)")
            plt.show()

    def epi_to_mni(self):
        inputs = [
            self.func_img_or_mc_bet_clean_withGSR,
            self.func_img_or_mc_bet_clean_withoutGSR,
           # self.t1_to_epi_mask, #! aussi amener le mask, on aura besoin pour zscore
           # self.func_img_or_mc_bet_mean #! je l'avais fait mais jsp pk ??? au cas ou il est la voir si ca prend trop de compute
        ]
        
        outputs = [
            self.func_img_or_mc_bet_clean_withGSR_mni,
            self.func_img_or_mc_bet_clean_withoutGSR_mni,
           # self.t1_to_epi_to_mni_mask,
           # self.func_img_or_mc_bet_mean_mni   
            ]
        for n,img in enumerate(inputs):
            out = outputs[n]
            if "mask" in str(img):
                interp = "nn"
            else:
                interp = "spline"
            applywarp(
                src= img,
                ref=MNI_HEAD_2mm,
                warp= self.anat_data.t1_to_mni_nnlin_warp,
                premat= self.epi_to_t1_mat,
                out = out,
                interp= interp,
            )

    def smoothing(self, fwhm = 4.0):
        inputs = [
            self.func_img_or_mc_bet_clean_withGSR_mni,
            self.func_img_or_mc_bet_clean_withoutGSR_mni
        ]
        outputs = [
            self.func_img_or_mc_bet_clean_withGSR_mni_smooth,
            self.func_img_or_mc_bet_clean_withoutGSR_mni_smooth
        ]
        for n,file in enumerate(inputs):
            out = outputs[n]
            fmri = nib.load(file)
            smoothed = nimg.smooth_img(fmri, fwhm)
            smoothed.to_filename(out)
    

    def visualize(self):

        sp.Popen(["fsleyes",
                    self.func_img_or,
                    self.func_img_or_mc,
                    self.func_img_or_mc_mean,
                    self.func_img_or_mc_bet,
                    self.func_img_or_mc_t1,
                    self.t1_to_epi_mask,
                    self.anat_data.t1_img_or,
                    self.anat_data.t1_img_or_brain,
                    MNI_HEAD_2mm,
                    self.func_img_or_mc_bet_clean_withGSR_mni_smooth,
                    self.func_img_or_mc_bet_clean_withoutGSR_mni_smooth,
                    #self.func_img_or_mc_bet_mean_mni,
                    self.anat_data.t1_img_or_brain_mask,
                  ])
        
    def run_func_pipeline(self):
        print(f"\n=== Running full functional pipeline for {self.sub_id} [{self.condition}] run {self.run} ===")

        t0 = time.perf_counter()

        step_start = time.perf_counter()
        print(f"  * [1/6] Reorient...", end="", flush=True)
        self.reorient()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [2/6] Motion correction...", end="", flush=True)
        with suppress_stdout_stderr():
            self.motion_correction()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [3/6] EPI to T1...", end="", flush=True)
        with suppress_stdout_stderr():
            self.func_to_t1()
            self.t1_mask_to_epi()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [4/6] Nuissance regression...", end="", flush=True)
        
        self.nuissance_regression(plot_design=True)
        print("OK")
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [5/6] EPI to MNI...", end="", flush=True)
        with suppress_stdout_stderr():
            self.epi_to_mni()
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        step_start = time.perf_counter()
        print(f"  * [6/6] Smoothing...", end="", flush=True)
        self.smoothing(fwhm=4.0)
        print(f" ✓ done ({time.perf_counter() - step_start:.1f}s)")

        total = time.perf_counter() - t0
        print(f"=== Done: {self.sub_id} [{self.condition}] run {self.run} ({total:.1f}s total) ===\n")
    
    
    
def get_full_subject_list():
    # Group 1 = control (Healthy)
    # Group 2 = diabetic (Type 2 diabetes)
    # Sex: F = Female, M = Male
    
    subjects = {
        'sub_001': {'group': 'control',  'sex': 'F'},
        'sub_002': {'group': 'diabetic', 'sex': 'M'},
        'sub_003': {'group': 'diabetic', 'sex': 'M'},
        'sub_004': {'group': 'diabetic', 'sex': 'M'},
        'sub_005': {'group': 'control',  'sex': 'M'},
        'sub_006': {'group': 'diabetic', 'sex': 'M'},
        'sub_007': {'group': 'diabetic', 'sex': 'F'},
        'sub_010': {'group': 'control',  'sex': 'M'},
        'sub_011': {'group': 'diabetic', 'sex': 'F'},
        'sub_012': {'group': 'control',  'sex': 'F'},
        'sub_013': {'group': 'control',  'sex': 'M'},
        'sub_015': {'group': 'control',  'sex': 'M'},
        'sub_017': {'group': 'diabetic', 'sex': 'M'},
        'sub_018': {'group': 'diabetic', 'sex': 'M'},
        'sub_019': {'group': 'diabetic', 'sex': 'F'},
        'sub_024': {'group': 'control',  'sex': 'M'},
        'sub_030': {'group': 'diabetic', 'sex': 'M'}
    }
    return subjects
    
    
class Subject:
    def __init__(self, sub_id : str):
        self.sub_id = sub_id
        data_hyper ={}
        data_hypo = {}
        data_hyper["anat"] = AnatT1(sub_id, "hyper")
        data_hypo["anat"] = AnatT1(sub_id, "hypo")
        data_hyper["func"] = {}
        data_hypo["func"] = {}
        for run in [f"R{i}" for i in range(1,5)]:
            data_hyper["func"][run] = funcFMRI(sub_id, "hyper", run)
            data_hypo["func"][run] = funcFMRI(sub_id, "hypo", run)
        self.data_hyper = data_hyper
        self.data_hypo = data_hypo
        self.group = get_full_subject_list()[sub_id]["group"]
        self.sex = get_full_subject_list()[sub_id]["sex"]
    # def get_hypo_run(self, run):
    #     return self.data_hypo["func"][run]
        
    # def get_hyper_run(self, run):
    #     return self.data_hyper["func"][run]

    def get_run(self, condition: str, run: str):
        if condition == "hyper":
            return self.data_hyper["func"][run]
        elif condition == "hypo":
            return self.data_hypo["func"][run]
        else:
            raise ValueError(f"Unknown condition {condition!r}")

    def get_processed_run(self, condition: str, run: str, gsr : bool):
        func_run = self.get_run(condition, run)
        if gsr:
            return func_run.func_img_or_mc_bet_clean_withGSR_mni_smooth
        else : 
            return func_run.func_img_or_mc_bet_clean_withoutGSR_mni_smooth
    
    # subj["hyper", "R1"]
    def __getitem__(self, key):
        condition, run = key
        return self.get_run(condition, run)




if __name__ == "__main__":
    pass

