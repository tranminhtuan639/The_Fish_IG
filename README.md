# Cấu trúc thư mục (md)
```
├── app
│   └── main.py
├── chatbot
│   ├── __init__.py
│   ├── llm_client.py
│   └── prompt_builder.py
├── data
│   └── knowledge_base
│       ├── arowana
│       │   ├── care.md
│       │   ├── food.md
│       │   └── variants.md
│       ├── flowerhorn
│       │   ├── care.md
│       │   ├── food.md
│       │   └── variants.md
│       ├── oscar
│       │   ├── care.md
│       │   ├── food.md
│       │   └── variants.md
│       └── snakehead
│           ├── care.md
│           ├── food.md
│           └── variants.md
├── model
│   ├── checkpoints
│   │   ├── best_model.pth
│   │   ├── best_phase1.pth
│   │   ├── best_phase2.pth
│   │   └── confusion_matrix.png
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── rag
│   ├── __init__.py
│   ├── build_index.py
│   └── retriever.py
├── utils
│   ├── __init__.py
│   └── split_dataset.py
├── .gitignore
├── README.md
├── check.py
└── requirements.txt
```
# Cấu trúc thư mục ( json)
```
"path": "c:\\",
  "type": "directory",
  "children": [
    {
      "name": "app",
      "type": "folder",
      "children": [
        {
          "name": "main.py",
          "type": "file"
        }
      ]
    },
    {
      "name": "chatbot",
      "type": "folder",
      "children": [
        {
          "name": "__init__.py",
          "type": "file"
        },
        {
          "name": "llm_client.py",
          "type": "file"
        },
        {
          "name": "prompt_builder.py",
          "type": "file"
        }
      ]
    },
    {
      "name": "data",
      "type": "folder",
      "children": [
        {
          "name": "knowledge_base",
          "type": "folder",
          "children": [
            {
              "name": "arowana",
              "type": "folder",
              "children": [
                {
                  "name": "care.md",
                  "type": "file"
                },
                {
                  "name": "food.md",
                  "type": "file"
                },
                {
                  "name": "variants.md",
                  "type": "file"
                }
              ]
            },
            {
              "name": "flowerhorn",
              "type": "folder",
              "children": [
                {
                  "name": "care.md",
                  "type": "file"
                },
                {
                  "name": "food.md",
                  "type": "file"
                },
                {
                  "name": "variants.md",
                  "type": "file"
                }
              ]
            },
            {
              "name": "oscar",
              "type": "folder",
              "children": [
                {
                  "name": "care.md",
                  "type": "file"
                },
                {
                  "name": "food.md",
                  "type": "file"
                },
                {
                  "name": "variants.md",
                  "type": "file"
                }
              ]
            },
            {
              "name": "snakehead",
              "type": "folder",
              "children": [
                {
                  "name": "care.md",
                  "type": "file"
                },
                {
                  "name": "food.md",
                  "type": "file"
                },
                {
                  "name": "variants.md",
                  "type": "file"
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "name": "model",
      "type": "folder",
      "children": [
        {
          "name": "checkpoints",
          "type": "folder",
          "children": [
            {
              "name": "best_model.pth",
              "type": "file"
            },
            {
              "name": "best_phase1.pth",
              "type": "file"
            },
            {
              "name": "best_phase2.pth",
              "type": "file"
            },
            {
              "name": "confusion_matrix.png",
              "type": "file"
            }
          ]
        },
        {
          "name": "evaluate.py",
          "type": "file"
        },
        {
          "name": "predict.py",
          "type": "file"
        },
        {
          "name": "train.py",
          "type": "file"
        }
      ]
    },
    {
      "name": "rag",
      "type": "folder",
      "children": [
        {
          "name": "__init__.py",
          "type": "file"
        },
        {
          "name": "build_index.py",
          "type": "file"
        },
        {
          "name": "retriever.py",
          "type": "file"
        }
      ]
    },
    {
      "name": "utils",
      "type": "folder",
      "children": [
        {
          "name": "__init__.py",
          "type": "file"
        },
        {
          "name": "split_dataset.py",
          "type": "file"
        }
      ]
    },
    {
      "name": ".gitignore",
      "type": "file"
    },
    {
      "name": "README.md",
      "type": "file"
    },
    {
      "name": "check.py",
      "type": "file"
    },
    {
      "name": "requirements.txt",
      "type": "file"
    }
  ],
  ```
```
The_Fish_IG/
├── app/          → Giao diện Streamlit (main.py)
├── chatbot/      → LLM client, Prompt builder
├── data/         → knowledge_base (Markdown), raw/processed images
├── model/        → Train, Predict, Evaluate + model checkpoints
├── rag/          → Build index, Retriever (ChromaDB)
└── utils/        → Tiện ích (split_dataset)
```
