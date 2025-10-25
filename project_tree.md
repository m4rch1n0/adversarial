adversarial_lab/
├─ pyproject.toml                   # Dependencies
├─ uv.lock                          # Lockfile dependencies (uv-compatible)
│
├─ configs/                         # Configurations experiments
│  ├─ vision/                      # Vision experiments (FGSM/PGD attacks)
│  ├─ poisoning/                   # Poisoning experiments (Dirty-label/Backdoor)
│  └─ llm/                         # LLM prompt injection experiments
│
├─ data/
│  ├─ imagenette2-160/              # Dataset Imagenette (10 classes ImageNet)
│  │  ├─ train/                     # Training set
│  │  └─ val/                       # Validation set
│  ├─ imagenette2-160.tgz           # Original archive
│  └─ llm/                          # LLM prompt injection test cases (24 total)
│     ├─ semantic_framing.json      # 6 RAG-based semantic framing attacks
│     ├─ obfuscation.json           # 6 obfuscation attacks (3 direct + 3 RAG)
│     ├─ jailbreak.json             # 6 jailbreak attacks (4 direct + 2 RAG)
│     ├─ clean.json                 # 6 clean baseline cases (4 direct + 2 RAG)
│     ├─ test.json                  # Temporary single-case testing
│     └─ documents/                 # External documents for RAG injection (13 total)
│        ├─ semantic_framing/       # 6 documents (PDF/TXT)
│        ├─ obfuscation/            # 3 documents (PDF/TXT)
│        ├─ jailbreak/              # 2 documents (PDF/TXT)
│        └─ clean/                  # 2 documents (PDF/TXT)
│
├─ results/                         # Output experiments (runs are timestamped)
│  ├─ vision/                       # FGSM/PGD attack results
│  ├─ training/                     # Poisoning experiments results
│  └─ llm/                          # LLM prompt injection results
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
│  ├─ training/
│  │  └─ poisoning.py               # PoisonedDataset, TriggeredDataset, label flipping
│  └─ llm/
│     ├─ agent.py                   # LLM agent with email tool, system prompts
│     ├─ dataset.py                 # Test case loader, RAG document handling
│     └─ eval.py                    # ASR, FAR metrics, response logging
│
└─ scripts/                         # Experiment runners
   ├─ run_experiment.sh             # experiments
   ├─ validate_config.py            # config validation helper
   ├─ run_vision.py                 # Train model + FGSM/PGD evaluation
   ├─ visualize_vision.py           # visualize adversarial examples grid
   └─ run_poisoning.py              # dirty-label/backdoor poisoning experiments