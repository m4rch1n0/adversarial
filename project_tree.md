adversarial_lab/
├─ pyproject.toml                   # Dependencies (torch, torchvision, numpy, matplotlib, tqdm)
├─ uv.lock                          # Lockfile dependencies (uv-compatible)
│
├─ configs/                         # Configurations experiments
│  ├─ vision/                      # Vision experiments (FGSM/PGD attacks)
│  └─ poisoning/                   # Poisoning experiments (Dirty-label/Backdoor)
│
├─ data/
│  ├─ imagenette2-160/              # Dataset Imagenette (10 classes ImageNet)
│  │  ├─ train/                     # Training set
│  │  └─ val/                       # Validation set
│  └─ imagenette2-160.tgz           # Original archive
│
├─ results/                         # Output experiments (runs are timestamped)
│  ├─ vision/                       # FGSM/PGD attack results
│  └─ training/                     # Poisoning experiments results
│
├─ src/aml_lab/                     # Core library
│  ├─ common/
│  │  ├─ seed.py                    # set_seed() - reproducibility
│  │  ├─ io.py                      # run_dir(), save_csv(), save_metadata()
│  │  └─ metrics.py                 # accuracy utilities
│  ├─ vision/
│  │  ├─ data.py                    # get_loaders() - ImageNet transforms
│  │  ├─ models.py                  # build_model() - CNN/ResNet18
│  │  ├─ attacks.py                 # fgsm(), pgd() - L-infinity attacks
│  │  └─ eval.py                    # evaluate() - clean/adversarial accuracy
│  └─ training/
│     └─ poisoning.py               # PoisonedDataset, TriggeredDataset, label flipping
│
└─ scripts/                         # Experiment runners
   ├─ run_experiment.sh             # experiments
   ├─ validate_config.py            # config validation helper
   ├─ run_vision.py                 # Train model + FGSM/PGD evaluation
   ├─ visualize_vision.py           # visualize adversarial examples grid
   └─ run_poisoning.py              # dirty-label/backdoor poisoning experiments