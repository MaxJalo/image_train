<!-- 🏛️ System architecture and component interactions -->
# System Architecture 

## Data Flow Diagram

```mermaid
graph TD
    A["📱 Client Application"] -->|"POST /process-folder"| B["🌐 FastAPI Server"]
    B -->|"Health Checks"| C["✅ Health Check Manager"]
    C -->|"DB Status"| Z["💾 MongoDB"]
    
    B -->|"Request handling"| D["⚙️ Processor Service"]
    D -->|"Discover images"| E["🖼️ Image Discovery<br/>get_all_images"]
    E -->|"Sorted list"| F["🔄 Sequential Processing"]
    
    F -->|"Image 1..N"| G["🤖 Model-1: Segmenter<br/>_predict_model1"]
    G -->|"'one_wagon' or<br/>'transition'"| H{"Wagon<br/>Boundary?"}
    
    H -->|"one_wagon"| I["📦 Add to current wagon"]
    H -->|"transition"| J["✂️ Finalize wagon<br/>Start new"]
    
    I -->|"Continue"| F
    J -->|"Process completed wagon"| K["🚂 _process_wagon"]
    
    K -->|"For each photo"| L["🤖 Model-2: Classifier<br/>_predict_model2"]
    L -->|"side + features"| M["📊 Aggregator<br/>Statistics"]
    M -->|"left_count,<br/>right_count"| N["🏆 Final Verdict<br/>Majority Vote"]
    
    N -->|"Wagon result"| O["💾 MongoDB<br/>PhotoDocument"]
    
    F -->|"All processed"| P["📋 Batch Summary"]
    P -->|"Save batch"| Q["💾 MongoDB<br/>BatchDocument"]
    
    Q -->|"Result ready"| R["✅ Return to Client"]
    R -->|"GET /result"| A
    
    style A fill:#e1f5ff
    style B fill:#0077be
    style E fill:#ffd700
    style G fill:#ff6b6b
    style L fill:#4ecdc4
    style M fill:#95e1d3
    style Z fill:#2ecc71
    style R fill:#90ee90
```

## Component Interaction

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Processor
    participant M1 as Model-1
    participant M2 as Model-2
    participant DB as MongoDB

    Client->>API: POST /process-folder<br/>folder_path
    API->>API: Check DB health
    API->>Processor: process_folder()
    
    Processor->>Processor: get_all_images()<br/>Sort by filename
    
    loop For each image
        Processor->>M1: _predict_model1(image)
        M1->>M1: Classify frame
        M1-->>Processor: "one_wagon"/"transition"
        
        alt "one_wagon"
            Processor->>Processor: Add to current_wagon
        else "transition"
            Processor->>Processor: _process_wagon(current_wagon)
            
            loop For each photo in wagon
                Processor->>M2: _predict_model2(photo)
                M2->>M2: Detect features<br/>Classify side
                M2-->>Processor: side, confidence, features
                
                Processor->>DB: Save PhotoDocument
                DB-->>Processor: ✅ Saved
            end
            
            Processor->>Processor: Aggregate statistics
            Processor->>DB: Save wagon result
        end
    end
    
    Processor->>DB: Save BatchDocument
    DB-->>Processor: ✅ Complete
    
    Processor-->>API: Results
    API-->>Client: {wagons, summary, status}
```

## Data Schema Relationships

```mermaid
erDiagram
    BATCH ||--o{ PHOTO : contains
    WAGON ||--o{ PHOTO : has
    
    BATCH {
        string batch_id PK
        string folder
        int total_photos
        int total_wagons
        int processed_photos
        string status
        datetime processed_at
        datetime created_at
        object results
    }
    
    WAGON {
        string wagon_id PK
        int total_photos
        int processed_photos
        int left_count
        int right_count
        string final_side
        array cameras
    }
    
    PHOTO {
        objectid _id PK
        string wagon_id FK
        string batch_id FK
        string file_hash
        int camera_id
        string side
        float confidence
        object features
        datetime processed_at
    }
    
    FEATURES {
        float brake_rod "0.0-1.0"
        float rod_nose "0.0-1.0"
        float crane "0.0-1.0"
        float tank "0.0-1.0"
    }
```

## Model-1: Wagon Segmentation

```mermaid
graph LR
    A["RGB Image<br/>224x224"] -->|"Preprocessing"| B["Normalized<br/>ImageNet"]
    B -->|"MobileNetV3Small"| C["Conv Layers<br/>3.17M params"]
    C -->|"Global Avg Pool"| D["Feature Vector<br/>1280-dim"]
    D -->|"Linear Layer"| E["Logits<br/>2-dim"]
    E -->|"Softmax"| F["Probabilities"]
    F -->|"argmax"| G{"Classification"}
    
    G -->|"Class 0"| H["✅ one_wagon<br/>Prob: 0.85"]
    G -->|"Class 1"| I["🔄 transition<br/>Prob: 0.72"]
    
    style A fill:#ffd700
    style C fill:#ff6b6b
    style H fill:#90ee90
    style I fill:#ffb6c6
```

## Model-2: Side Classifier

```mermaid
graph LR
    A["RGB Image<br/>Variable size"] -->|"Resize"| B["Fixed size<br/>Input"]
    B -->|"YOLO Backbone"| C["Feature Maps"]
    C -->|"Multi-scale<br/>Detection"| D["Bounding Boxes<br/>+ Classes"]
    D -->|"NMS"| E["Filtered Boxes"]
    E -->|"Feature Extraction"| F["Component<br/>Detections"]
    
    F -->|"brake_rod<br/>confidence"| G["📊 Decision<br/>Logic"]
    F -->|"rod_nose<br/>confidence"| G
    F -->|"crane<br/>confidence"| G
    F -->|"tank<br/>confidence"| G
    
    G -->|"Side heuristic<br/>+ all votes"| H["left_count<br/>right_count"]
    H -->|"Majority"| I["Final Side<br/>left/right"]
    
    style A fill:#ffd700
    style F fill:#4ecdc4
    style I fill:#90ee90
```

## Processing Pipeline State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing: start process_folder
    
    Initializing --> ImageDiscovery: ensure_db_connection
    ImageDiscovery --> Sorting: get_all_images
    Sorting --> Sequential: sort by filename
    
    Sequential --> Wagon1Start: wagon_count = 1
    Wagon1Start --> Processing: current_wagon = []
    
    Processing --> Model1: for each image
    Model1 --> Check: Run _predict_model1
    
    Check -->|"one_wagon"| AddPhoto: append to current_wagon
    AddPhoto --> Processing: continue loop
    
    Check -->|"transition"| SaveWagon: _process_wagon
    SaveWagon --> RunModel2: for each photo
    RunModel2 --> Aggregate: _predict_model2
    Aggregate --> Statistics: count left/right
    Statistics --> VerdictDB: save to MongoDB
    VerdictDB --> NewWagon: wagon_count++
    
    NewWagon --> Processing: resume from next image
    
    Processing --> FinalWagon: end of images
    FinalWagon --> SaveFinal: _process_wagon(last)
    SaveFinal --> Complete: Save BatchDocument
    
    Complete --> [*]
    
    note right of Check
        Critical: Image order matters!
        Determines wagon boundaries
    end note
    
    note right of Aggregate
        Each photo → Model-2 inference
        Extract all features
    end note
    
    note right of Statistics
        Count left/right classifications
        Determine final_side by majority
    end note
```

## Error Handling Flow

```mermaid
graph TD
    A["Start Processing"] --> B["ensure_db_connection"]
    B -->|DB OK| C["Continue Processing"]
    B -->|DB Failed| D["Log Warning<br/>db_ok=False"]
    
    C --> E["Process Images"]
    D --> E
    
    E --> F{"Model Error?"}
    F -->|Yes| G["Log Error<br/>Skip Photo"]
    F -->|No| H["Save Result"]
    
    G --> I{"Continue?"}
    I -->|Yes| E
    I -->|No| J["Error Batch Doc"]
    
    H --> K{"DB Available?"}
    K -->|Yes| L["Save to MongoDB"]
    K -->|No| M["In-Memory Only"]
    
    L --> N{"Save OK?"}
    N -->|Yes| O["✅ Photo Saved"]
    N -->|No| P["❌ Log Error<br/>Continue"]
    
    M --> O
    P --> I
    
    O --> Q{"More Images?"}
    Q -->|Yes| E
    Q -->|No| R["Batch Complete"]
    
    R --> S{"DB Available?"}
    S -->|Yes| T["Save BatchDoc"]
    S -->|No| U["Return In-Memory"]
    
    T --> V["✅ Success"]
    U --> V
    J --> V
    
    style V fill:#90ee90
    style P fill:#ffb6c6
    style U fill:#fff9c4
```

## Deployment Architecture

```mermaid
graph TB
    Client["📱 Client"]
    LB["⚖️ Load Balancer<br/>Nginx"]
    API1["🌐 API Pod 1<br/>FastAPI"]
    API2["🌐 API Pod 2<br/>FastAPI"]
    API3["🌐 API Pod 3<br/>FastAPI"]
    
    Cache["⚡ Cache<br/>Redis"]
    Queue["📋 Queue<br/>Celery"]
    
    DB["💾 MongoDB<br/>Replica Set"]
    Model1["🤖 Model-1<br/>GPU Pod"]
    Model2["🤖 Model-2<br/>GPU Pod"]
    
    Storage["📦 S3/MinIO<br/>Object Storage"]
    
    Client --> LB
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> Cache
    API2 --> Cache
    API3 --> Cache
    
    API1 --> Queue
    API2 --> Queue
    API3 --> Queue
    
    API1 --> DB
    API2 --> DB
    API3 --> DB
    
    API1 --> Model1
    API2 --> Model2
    API3 --> Model1
    
    Model1 --> Storage
    Model2 --> Storage
    
    Queue -.->|Worker| Model1
    Queue -.->|Worker| Model2
    
    style API1 fill:#0077be
    style API2 fill:#0077be
    style API3 fill:#0077be
    style DB fill:#2ecc71
    style Model1 fill:#ff6b6b
    style Model2 fill:#4ecdc4
    style Cache fill:#fff9c4
```

---

## Call Flow Examples

### Successful Processing

```
1. Client sends folder path
   POST /process-folder {folder: "/test/camera_2"}

2. FastAPI receives request
   - Checks /health/db → Connected ✅
   - Spawns processor task

3. Processor starts
   - ensure_db_connection() → True
   - get_all_images() → [image1, image2, ..., image_N] (sorted)
   - Creates batch_id

4. For each image:
   - Model-1: "one_wagon" (frames 1-10)
   - Model-1: "one_wagon" (frames 11-20)
   - Model-1: "transition" (frame 21)
     → Triggers _process_wagon("wagon_1", frames_1-20)
   - Model-2: side=left (frame 1) → left_count++
   - Model-2: side=right (frame 2) → right_count++
   - ... aggregate all 20 frames
   - Save wagon_1 result + PhotoDocuments

5. Continue with wagon_2, wagon_3...

6. All wagons processed:
   - Save BatchDocument with all results
   - Return to client: {batch_id, wagons, summary, status}

7. Client polls GET /result/{batch_id}
   - Returns complete results from MongoDB
```

### DB Unavailable Scenario

```
1. Client sends folder path
   POST /process-folder {folder: "/test"}

2. FastAPI receives request
   - Checks /health/db → NOT Connected ⚠️
   - Sets db_ok=False
   - But continues processing!

3. Processor processes all images
   - All computations work normally

4. When saving results:
   - If db_ok=False: Skip MongoDB save
   - Results stored in memory
   - Return to client with results

5. Client gets full results without DB
   - Can retry later if DB comes back online
   - Results still valid but not persisted
```

---

## Performance Characteristics

```mermaid
graph LR
    A["Image Count"] -->|"× Time per Image"| B["Total Time"]
    
    C["Model-1 Time<br/>50-100ms"] -->|"+ Model-2 Time"| D["Per Image: 250-400ms"]
    E["Model-2 Time<br/>200-300ms"] --> D
    
    F["100 Images"] -->|"5-8 min"| G["Typical"]
    H["500 Images"] -->|"30-45 min"| G
    I["1000 Images"] -->|"60-90 min"| G
    
    J["GPU (CUDA)"] -->|"3-5x faster"| K["GPU Processing"]
    L["CPU Only"] -->|"1x baseline"| K
    
    M["Parallelization"] -->|"N cameras"| N["Multi-stream"]
    
    style G fill:#ffd700
    style K fill:#90ee90
```

---

*Architecture diagrams created: February 24, 2026*
