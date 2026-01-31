# Granite-8B Training Pipeline - State Machine Diagram

## 1. Top-Level Pipeline State Machine

```mermaid
stateDiagram-v2
    [*] --> Validation
    Validation --> DataPreparation : All checks pass
    Validation --> [*] : Validation failed

    DataPreparation --> TeacherGeneration : train.jsonl + val.jsonl ready
    DataPreparation --> [*] : No data extracted

    TeacherGeneration --> TrainingLaunch : Teacher outputs generated
    TeacherGeneration --> TrainingLaunch : Skip (use raw data only)

    TrainingLaunch --> IterativeDistillation : SageMaker job running
    TrainingLaunch --> [*] : Launch failed

    IterativeDistillation --> Complete : Converged or max epochs
    IterativeDistillation --> [*] : Fatal error

    Complete --> [*] : Model saved to S3

    state Validation {
        [*] --> AWSCredentials
        AWSCredentials --> S3Access : Valid
        AWSCredentials --> ValidationError : Invalid
        S3Access --> BedrockAuth : Bucket accessible
        S3Access --> ValidationError : Access denied
        BedrockAuth --> IAMRole : Model responds
        BedrockAuth --> ValidationWarn : --skip-bedrock
        IAMRole --> HFToken : Role exists
        IAMRole --> ValidationError : Role missing
        HFToken --> ConfigCheck : Token valid
        HFToken --> ValidationWarn : Token missing
        ConfigCheck --> LocalFiles : YAML valid
        ConfigCheck --> ValidationError : Invalid config
        LocalFiles --> ScriptSyntax : Files exist
        ScriptSyntax --> ImportCheck : Valid Python
        ImportCheck --> [*] : All imports resolve
        ValidationError --> [*]
        ValidationWarn --> ConfigCheck
    }

    state DataPreparation {
        [*] --> InitDirs
        InitDirs --> S3Download
        S3Download --> TSNProcess : TSN files downloaded
        TSNProcess --> AVBProcess : Functions extracted
        AVBProcess --> CARLAProcess : AVB data processed
        CARLAProcess --> Deduplication : CARLA data processed
        Deduplication --> TrainValSplit : Duplicates removed
        TrainValSplit --> SaveSplits : 90/10 split
        SaveSplits --> [*] : JSONL files saved
    }

    state TeacherGeneration {
        [*] --> InitBedrock
        InitBedrock --> LoadPrompts : Client ready
        LoadPrompts --> ParallelGenerate : Prompts loaded
        ParallelGenerate --> CheckpointSave : Batch complete
        CheckpointSave --> ParallelGenerate : More prompts
        CheckpointSave --> [*] : All done
    }

    state IterativeDistillation {
        [*] --> LoadModel
        LoadModel --> EpochLoop : Granite-8B + QLoRA ready

        state EpochLoop {
            [*] --> TrainStep
            TrainStep --> GenerateOutputs : Loss computed
            GenerateOutputs --> EvaluateQuality : Student outputs ready
            EvaluateQuality --> IdentifyPoor : Scores calculated
            IdentifyPoor --> GetCorrections : Poor outputs found
            IdentifyPoor --> ConvergenceCheck : No poor outputs
            GetCorrections --> AddToDataset : Teacher corrections received
            AddToDataset --> ConvergenceCheck : Dataset updated
            ConvergenceCheck --> [*] : Converged
            ConvergenceCheck --> TrainStep : Not converged
        }

        EpochLoop --> SaveFinalModel : Loop complete
        SaveFinalModel --> [*]
    }
```

## 2. Data Preparation Detail

```mermaid
stateDiagram-v2
    [*] --> CreateDirectories
    CreateDirectories --> InitS3Client

    state S3Download {
        [*] --> ListTSNObjects
        ListTSNObjects --> DownloadTSN : Objects found
        ListTSNObjects --> ListAVBObjects : No TSN data
        DownloadTSN --> ListAVBObjects : Complete
        ListAVBObjects --> DownloadAVB : Objects found
        ListAVBObjects --> ListCARLA : No AVB data
        DownloadAVB --> ListCARLA : Complete
        ListCARLA --> DownloadCARLA : Objects found
        DownloadCARLA --> [*] : Complete
    }

    InitS3Client --> S3Download

    state ProcessFiles {
        [*] --> ReadFile
        ReadFile --> DecodeUTF8
        DecodeUTF8 --> ExtractFunctions : Success
        DecodeUTF8 --> DecodeLatin1 : UnicodeDecodeError
        DecodeLatin1 --> ExtractFunctions : Success
        DecodeLatin1 --> SkipFile : Failed

        ExtractFunctions --> GeneratePrompts : Functions found
        ExtractFunctions --> GenerateCodeCompletion : No functions (len > 200)
        ExtractFunctions --> SkipFile : Too short

        GeneratePrompts --> SelectTemplate : Per function
        SelectTemplate --> FormatOutput : Template chosen (4 variants)
        GenerateCodeCompletion --> SplitCode : Random split point
        SplitCode --> FormatOutput

        FormatOutput --> [*] : Example added
        SkipFile --> [*]
    }

    S3Download --> ProcessFiles

    state Finalize {
        [*] --> DeduplicateMD5
        DeduplicateMD5 --> ShuffleSeed42
        ShuffleSeed42 --> SplitAt90Pct
        SplitAt90Pct --> WriteTrainJSONL
        SplitAt90Pct --> WriteValJSONL
        WriteTrainJSONL --> [*]
        WriteValJSONL --> [*]
    }

    ProcessFiles --> Finalize
    Finalize --> [*]
```

## 3. Teacher Output Generation Detail

```mermaid
stateDiagram-v2
    [*] --> InitBedrockClient

    state Authentication {
        [*] --> CheckIAMCreds
        CheckIAMCreds --> CheckAPIKey : IAM valid
        CheckAPIKey --> RegisterEventHandler : API key found
        CheckAPIKey --> ClientReady : No API key (IAM only)
        RegisterEventHandler --> ClientReady : Header hook registered
    }

    InitBedrockClient --> Authentication
    Authentication --> LoadPrompts

    LoadPrompts --> ValidatePrompts : JSONL loaded
    LoadPrompts --> CreateTestPrompts : --test mode

    ValidatePrompts --> CreateThreadPool : Valid prompts
    CreateTestPrompts --> CreateThreadPool

    state ParallelExecution {
        [*] --> SubmitFutures
        SubmitFutures --> ProcessFuture

        state ProcessFuture {
            [*] --> InvokeModel
            InvokeModel --> ParseResponse : 200 OK
            InvokeModel --> RetryBackoff : ThrottlingException
            InvokeModel --> RetryBackoff : ModelStreamError
            InvokeModel --> RecordFailure : Other error
            InvokeModel --> RecordFailure : MaxRetries exceeded

            RetryBackoff --> InvokeModel : sleep(base * 2^attempt)
            ParseResponse --> RecordSuccess : Text extracted
            RecordSuccess --> [*]
            RecordFailure --> [*]
        }

        ProcessFuture --> CheckpointWrite : Every 10 results
        CheckpointWrite --> ProcessFuture : More futures
        ProcessFuture --> [*] : All complete
    }

    CreateThreadPool --> ParallelExecution
    ParallelExecution --> SortAndSave : All results collected
    SortAndSave --> PrintSummary
    PrintSummary --> [*]
```

## 4. Iterative Distillation Epoch Loop

```mermaid
stateDiagram-v2
    [*] --> DetectEnvironment

    state Setup {
        DetectEnvironment --> SageMakerMode : SM_TRAINING_ENV set
        DetectEnvironment --> LocalMode : Not on SageMaker
        SageMakerMode --> CreateDirs
        LocalMode --> CreateDirs
        CreateDirs --> LoadTrainData
        LoadTrainData --> LoadEvalPrompts
        LoadEvalPrompts --> LoadGranite8B
    }

    state ModelSetup {
        LoadGranite8B --> ApplyQuantization : 4-bit nf4
        ApplyQuantization --> PrepareKBit
        PrepareKBit --> ApplyLoRA : rank=32, alpha=64
        ApplyLoRA --> InitTokenizer
        InitTokenizer --> InitEvaluator
        InitEvaluator --> InitTeacher : CodeQualityEvaluator
        InitTeacher --> CreateSFTTrainer : BedrockTeacherGenerator
    }

    Setup --> ModelSetup
    ModelSetup --> EpochN

    state EpochN {
        [*] --> Train
        note right of Train
            SFTTrainer.train()
            on train_dataset + corrections
        end note

        Train --> Generate : train_loss recorded
        note right of Generate
            model.eval()
            For each eval prompt:
            - Tokenize (max 4096)
            - Generate (max_new_tokens=1024)
            - temperature=0.7, top_p=0.95
        end note

        Generate --> Evaluate : student_outputs[]
        note right of Evaluate
            Per output:
            - syntax_score (GCC check)
            - protocol_score (TSN/AVB keywords)
            - safety_score (MISRA-C)
            - style_score (comments/docs)
            Overall = 0.3*syn + 0.3*proto + 0.25*safe + 0.15*style
        end note

        Evaluate --> FilterPoor : scores[]
        FilterPoor --> RequestCorrections : score < 7.0
        FilterPoor --> CheckConvergence : All scores >= 7.0

        state RequestCorrections {
            [*] --> SortByScore
            SortByScore --> CapAt500 : Worst first
            CapAt500 --> ParallelTeacherCalls
            ParallelTeacherCalls --> FormatAsTraining
            FormatAsTraining --> AppendToDataset
            AppendToDataset --> [*]
        }

        RequestCorrections --> CheckConvergence

        state CheckConvergence {
            [*] --> CheckThreshold
            CheckThreshold --> Converged : avg >= 8.0 for 3+ epochs
            CheckThreshold --> CheckCorrectionRate : Not at threshold
            CheckCorrectionRate --> Converged : rate < 10% for 3 epochs
            CheckCorrectionRate --> Continue : Still improving
            Converged --> [*]
            Continue --> [*]
        }

        CheckConvergence --> [*] : Converged
        CheckConvergence --> Train : Continue (next epoch)
    }

    EpochN --> SaveModel : Converged or max epochs
    SaveModel --> SaveMetrics
    SaveMetrics --> [*]
```

## 5. QLoRA Training Loop (SageMaker)

```mermaid
stateDiagram-v2
    [*] --> ParseHyperparams

    state Initialization {
        ParseHyperparams --> LoadDatasets : From SM_CHANNEL_*
        LoadDatasets --> ValidateStructure : Check messages field
        ValidateStructure --> CheckDiskSpace : >= 15GB free
        CheckDiskSpace --> CreateQuantConfig
    }

    state ModelLoading {
        CreateQuantConfig --> LoadGranite8B : BitsAndBytes 4-bit nf4
        LoadGranite8B --> PrepareForKBit
        PrepareForKBit --> ApplyLoRAAdapters
        ApplyLoRAAdapters --> LoadTokenizer
        LoadTokenizer --> SetPadToken : pad = eos
    }

    Initialization --> ModelLoading

    state TrainerSetup {
        SetPadToken --> CreateTrainingArgs
        CreateTrainingArgs --> CreateCallbacks
        CreateCallbacks --> CreateSFTTrainer

        state CreateCallbacks {
            [*] --> NaNDetector
            NaNDetector --> EarlyStopCallback
            EarlyStopCallback --> [*]
        }
    }

    ModelLoading --> TrainerSetup

    state TrainingLoop {
        [*] --> ForwardPass
        ForwardPass --> ComputeLoss
        ComputeLoss --> CheckNaN

        CheckNaN --> BackwardPass : Loss valid
        CheckNaN --> HaltTraining : NaN or Inf detected

        BackwardPass --> AccumulateGradients
        AccumulateGradients --> ForwardPass : accum < 8
        AccumulateGradients --> OptimizerStep : accum == 8

        OptimizerStep --> LogMetrics : Every 10 steps
        LogMetrics --> EvalCheck : Every 50 steps?

        state EvalCheck {
            [*] --> RunValidation
            RunValidation --> CompareEvalLoss
            CompareEvalLoss --> UpdateBest : New best
            CompareEvalLoss --> IncrementPatience : No improvement
            UpdateBest --> [*]
            IncrementPatience --> EarlyStop : patience >= 3
            IncrementPatience --> [*] : patience < 3
        }

        EvalCheck --> SaveCheckpoint : Every 100 steps
        EvalCheck --> ForwardPass : Continue
        SaveCheckpoint --> ForwardPass

        EarlyStop --> [*] : Stop training
        HaltTraining --> [*] : Emergency stop
    }

    TrainerSetup --> TrainingLoop

    state SaveOutputs {
        [*] --> SaveAdapters
        SaveAdapters --> SaveTokenizer
        SaveTokenizer --> SaveConfig
        SaveConfig --> SaveMetricsJSON
        SaveMetricsJSON --> [*]
    }

    TrainingLoop --> SaveOutputs
    SaveOutputs --> [*]
```

## 6. Quality Scoring State Machine

```mermaid
stateDiagram-v2
    [*] --> ExtractCode

    state ExtractCode {
        [*] --> CheckMarkdown
        CheckMarkdown --> ParseCodeBlocks : Has ``` blocks
        CheckMarkdown --> UseRawText : No markdown
        ParseCodeBlocks --> [*]
        UseRawText --> [*]
    }

    ExtractCode --> SyntaxCheck

    state SyntaxCheck {
        [*] --> TryGCC
        TryGCC --> GCCCompile : GCC available
        TryGCC --> HeuristicCheck : GCC not found

        GCCCompile --> Score10 : Clean compile
        GCCCompile --> Score7to9 : Warnings only
        GCCCompile --> Score5to6 : Errors present
        GCCCompile --> Score0 : Multiple critical errors

        HeuristicCheck --> CheckBraces
        CheckBraces --> CheckParens
        CheckParens --> CheckSemicolons
        CheckSemicolons --> [*] : Base 7.0 +/- penalties
    }

    SyntaxCheck --> ProtocolCheck

    state ProtocolCheck {
        [*] --> DetectDomain
        DetectDomain --> CheckTSNKeywords : TSN prompt
        DetectDomain --> CheckAVBKeywords : AVB prompt
        DetectDomain --> GenericScore : Unknown domain

        CheckTSNKeywords --> CountMatches : pcp, vlan, timestamp, gate, priority, qbv, qav, shaper, gcl, schedule
        CheckAVBKeywords --> CountMatches : stream, sample, channel, bandwidth, srp, talker, listener, reservation

        CountMatches --> ApplyPenalties : < 3 keywords found
        CountMatches --> [*] : >= 3 keywords found
        ApplyPenalties --> [*] : -1.5 per missing keyword

        GenericScore --> [*] : Default 6.0
    }

    ProtocolCheck --> SafetyCheck

    state SafetyCheck {
        [*] --> CheckViolations
        CheckViolations --> GotoCheck : -3.0
        CheckViolations --> MallocCheck : -2.0 each
        CheckViolations --> SetjmpCheck : -3.0
        CheckViolations --> AbortCheck : -1.5

        GotoCheck --> CheckGoodPractices
        MallocCheck --> CheckGoodPractices
        SetjmpCheck --> CheckGoodPractices
        AbortCheck --> CheckGoodPractices

        CheckGoodPractices --> FixedWidthTypes : +0.5 each
        FixedWidthTypes --> StaticConst : +0.3 each
        StaticConst --> Volatile : +0.2

        Volatile --> CheckRecursion : -1.5 if found
        CheckRecursion --> CheckInfiniteLoops : -2.0 if while(1) no break
        CheckInfiniteLoops --> [*] : Clamped [0, 10]
    }

    SafetyCheck --> StyleCheck

    state StyleCheck {
        [*] --> CountComments
        CountComments --> Score8 : 5+ comments
        CountComments --> Score7 : 2-5 comments
        CountComments --> Score5 : 0 comments

        Score8 --> CheckDoxygen
        Score7 --> CheckDoxygen
        Score5 --> CheckDoxygen

        CheckDoxygen --> CheckVarNames : +0.5 to +2.0
        CheckVarNames --> CheckIndentation : -1.0 if >5 single-letter
        CheckIndentation --> [*] : -0.5 if mixed tabs/spaces
    }

    StyleCheck --> ComputeOverall

    state ComputeOverall {
        [*] --> WeightedAverage
        note right of WeightedAverage
            overall = 0.30 * syntax
                    + 0.30 * protocol
                    + 0.25 * safety
                    + 0.15 * style
        end note
        WeightedAverage --> NeedsCorrection : overall < 7.0
        WeightedAverage --> Acceptable : overall >= 7.0
    }

    ComputeOverall --> [*]
```

## 7. Data Flow Diagram

```mermaid
flowchart TD
    subgraph S3["S3: granite-8b-unified-automotive-data"]
        RAW["Raw Code\ntsn_data/ avb_data/ carla/"]
        SPLITS["data/splits/\ntrain.jsonl + val.jsonl"]
        PROCESSED["data/processed/\ntrain.jsonl + val.jsonl"]
        OUTPUTS["output/distillation/\nmodel + metrics"]
    end

    subgraph DataPrep["Phase 2: Data Preparation"]
        DOWNLOAD["Download from S3"]
        EXTRACT["Extract Functions\n(C/C++ regex)"]
        PROMPTS["Generate Prompts\n(4 templates)"]
        DEDUP["Deduplicate (MD5)"]
        SPLIT["Train/Val Split\n(90/10, seed=42)"]
    end

    subgraph TeacherGen["Phase 3: Teacher Generation"]
        BEDROCK["Claude Sonnet 4.5\n(Bedrock)"]
        PARALLEL["ThreadPool\n(10 workers)"]
        CHECKPOINT["Checkpoint\n(every 10 results)"]
    end

    subgraph SageMaker["SageMaker Training Job"]
        subgraph Container["HuggingFace DLC\nPyTorch 2.5.1 + CUDA 12.4"]
            MODEL["Granite-8B\n+ BitsAndBytes 4-bit"]
            LORA["LoRA Adapters\nr=32, alpha=64"]
            TRAINER["SFTTrainer\n+ Callbacks"]
        end

        subgraph DistillLoop["Iterative Distillation Loop"]
            TRAIN["1. Train Student"]
            GEN["2. Generate Outputs"]
            EVAL["3. Evaluate Quality"]
            CORRECT["4. Get Corrections"]
        end
    end

    RAW --> DOWNLOAD
    DOWNLOAD --> EXTRACT --> PROMPTS --> DEDUP --> SPLIT
    SPLIT --> SPLITS
    SPLIT --> PROCESSED

    SPLITS --> PARALLEL
    PARALLEL --> BEDROCK
    BEDROCK --> CHECKPOINT

    PROCESSED -->|SM_CHANNEL_TRAIN| Container
    Container --> TRAIN
    TRAIN --> GEN --> EVAL --> CORRECT
    CORRECT -->|"Teacher corrections\n(Bedrock API)"| TRAIN
    CORRECT --> OUTPUTS

    EVAL -->|"score >= 8.0\n3 consecutive"| OUTPUTS
```

## 8. Error Recovery State Machine

```mermaid
stateDiagram-v2
    state "Error Recovery Paths" as errors {

        state AWSErrors {
            [*] --> NoCredentials
            NoCredentials --> SetEnvVars : ACTION: Set AWS_ACCESS_KEY_ID
            [*] --> AccessDenied
            AccessDenied --> UpdateBucketPolicy : ACTION: Update S3 policy
            [*] --> RoleNotFound
            RoleNotFound --> DeployCloudFormation : ACTION: Create IAM role
        }

        state BedrockErrors {
            [*] --> ThrottlingException
            ThrottlingException --> ExponentialBackoff : sleep(1s * 2^attempt)
            ExponentialBackoff --> RetryInvoke : attempt < 5
            ExponentialBackoff --> SkipPrompt : attempt >= 5
            [*] --> ModelStreamError
            ModelStreamError --> ExponentialBackoff
        }

        state TrainingErrors {
            [*] --> NaNLoss
            NaNLoss --> LogCUDAMemory
            LogCUDAMemory --> HaltTraining : Immediate stop
            [*] --> CUDAOutOfMemory
            CUDAOutOfMemory --> ReduceBatchSize : Restart with smaller batch
            [*] --> DiskSpaceLow
            DiskSpaceLow --> CleanCheckpoints : Remove old checkpoints
        }

        state DataErrors {
            [*] --> EncodingError
            EncodingError --> TryLatin1 : Fallback encoding
            TryLatin1 --> SkipFile : Still fails
            [*] --> EmptyDataset
            EmptyDataset --> CheckS3Paths : Verify data exists
            [*] --> InvalidStructure
            InvalidStructure --> ValidateJSONL : Check messages field
        }

        state SageMakerErrors {
            [*] --> SpotInterruption
            SpotInterruption --> ResumeFromCheckpoint : Automatic restart
            [*] --> MaxRuntimeExceeded
            MaxRuntimeExceeded --> SavePartialModel : Save current state
            [*] --> ContainerError
            ContainerError --> CheckDLCImage : Verify image tag
        }
    }
```

## 9. Convergence Decision Logic

```mermaid
flowchart TD
    START["Epoch Complete"] --> CALC["Calculate Metrics\navg_score, correction_rate"]

    CALC --> CHECK1{"avg_score >= 8.0?"}
    CHECK1 -->|Yes| INC1["Increment epochs_at_threshold"]
    CHECK1 -->|No| RESET1["Reset epochs_at_threshold = 0"]

    INC1 --> CHECK2{"epochs_at_threshold >= 3?"}
    CHECK2 -->|Yes| CONVERGED1["CONVERGED\nReason: Quality threshold met\nfor 3 consecutive epochs"]
    CHECK2 -->|No| CHECK3

    RESET1 --> CHECK3{"correction_rate < 10%?"}
    CHECK3 -->|Yes| INC2["Increment low_correction_count"]
    CHECK3 -->|No| RESET2["Reset low_correction_count = 0"]

    INC2 --> CHECK4{"low_correction_count >= 3?"}
    CHECK4 -->|Yes| CONVERGED2["CONVERGED\nReason: Correction rate < 10%\nfor 3 consecutive epochs"]
    CHECK4 -->|No| CONTINUE

    RESET2 --> CHECK5{"epoch >= max_epochs?"}
    CHECK5 -->|Yes| MAXED["COMPLETE\nReason: Max epochs reached"]
    CHECK5 -->|No| CONTINUE["CONTINUE\nStart next epoch"]

    CONTINUE --> START
```

## 10. SageMaker Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Pending : create_training_job()

    Pending --> Starting : Resources allocated
    Starting --> Downloading : Container pulling image
    Downloading --> Training : Data channels downloaded
    Training --> Uploading : Training complete
    Uploading --> Completed : Model artifacts uploaded

    Training --> Failed : Runtime error
    Training --> Failed : MaxRuntimeExceeded
    Training --> Stopping : User requested stop
    Training --> Stopping : Spot interruption
    Stopping --> Stopped : Cleanup complete

    Completed --> [*] : model.tar.gz in S3
    Failed --> [*] : Check FailureReason
    Stopped --> [*] : Partial results saved

    state Training {
        [*] --> InstallDeps : pip install test deps
        InstallDeps --> PrintEnv : Log GPU/CUDA info
        PrintEnv --> RunPytest : Execute test suite
        RunPytest --> SaveResults : Tests complete
        SaveResults --> [*] : Results to /opt/ml/output
    }
```

## Summary of States by Phase

| Phase | Script | States | Decision Points | Error States |
|-------|--------|--------|-----------------|--------------|
| Validation | dry_run_pipeline.py | 9 | 9 | 6 |
| Data Prep | prepare_automotive_data.py | 10 | 4 | 4 |
| Teacher Gen | generate_teacher_outputs.py | 7 | 3 | 5 |
| Training Launch | run_training_job.py | 8 | 2 | 4 |
| Iterative Distillation | iterative_distillation.py | 12 | 3 | 5 |
| QLoRA Training | train_granite_qlora.py | 14 | 4 | 7 |
| Cloud Launch | launch_cloud_training.py | 9 | 2 | 4 |
| **Total** | **7 scripts** | **69 states** | **27 decision points** | **35 error states** |
