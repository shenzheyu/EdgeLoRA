# Compile server

```
cd llama.cpp
make -j8 llama-server
```

# Start server

```
./llama-server -m [model_file_path] -c 2048 --lora_repeated [lora_file_path] 20 --adapter_cache_size 10
```

# API example

```
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128,"adapter_idx": 1}'
```

or 

```
cd llama.cpp/llama-client
node index.js
```