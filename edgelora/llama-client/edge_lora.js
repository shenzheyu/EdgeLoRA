const gamma = require('gamma'); // Library to generate Gamma distributed random numbers
const ProgressBar = require('progress');

// Configuration parameters
const n = 50; // Number of adapters
const alpha = 1; // Power-law exponent
const R = 0.5; // Total request rate in requests per second
const cv = 1.5; // Coefficient of variance
const traceDuration = 1 * 60 * 1000; // 5 minutes in milliseconds
const Il = 8, Iu = 128; // Input length bounds
const Ol = 8, Ou = 256; // Output length bounds

let completedRequests = 0;

function generateRequests(params) {
    const {
        numAdapters,
        alpha,
        reqRate,
        cv,
        duration,
        inputRange,
        outputRange,
        seed
    } = params;

    // Calculate total requests
    const totalRequests = Math.floor(reqRate * duration);
    
    // Generate adapter IDs using power law distribution
    const adapterIds = Array(totalRequests).fill(0).map(() => {
        const prob = Math.random(); // Generates uniform random number [0,1)
        const powerLaw = Math.pow(prob, 1/alpha); // Transform to power law
        return Math.floor(powerLaw * numAdapters);
    });

    // Generate input and output lengths
    const inputLengths = Array(totalRequests).fill(0).map(() => 
        Math.floor(Math.random() * (inputRange[1] - inputRange[0] + 1)) + inputRange[0]
    );
    const outputLengths = Array(totalRequests).fill(0).map(() => 
        Math.floor(Math.random() * (outputRange[1] - outputRange[0] + 1)) + outputRange[0]
    );

    // Generate intervals using gamma distribution
    const shape = 1 / (cv * cv);
    const scale = cv * cv / reqRate;
    const intervals = Array(totalRequests).fill(0).map(() => 
        gamma(shape) * scale * 1000 // Convert to milliseconds
    );

    // Create requests with timestamps
    let timestamp = 0;
    const requests = [];
    
    for (let i = 0; i < totalRequests; i++) {
        timestamp += intervals[i];
        requests.push({
            id: i,
            time: timestamp,
            adapter_id: adapterIds[i],
            inputLength: inputLengths[i],
            outputLength: outputLengths[i]
        });
    }

    return requests;
}

async function setLoraAdapterScales(adapter_id, numAdapters) {
    // Create array of adapter configs
    const adapterConfigs = Array(numAdapters).fill(0).map((_, idx) => ({
        id: idx,
        scale: idx === adapter_id ? 1.0 : 0.0
    }));

    // Call LoRA adapters endpoint
    try {
        const response = await fetch("http://127.0.0.1:8080/lora-adapters", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(adapterConfigs)
        });

        // console.log(JSON.stringify(adapterConfigs))
        
        if (!response.ok) {
            throw new Error(`Failed to set LoRA adapters: ${response.statusText}`);
        }
    } catch (error) {
        console.error('Error setting LoRA adapters:', error);
        throw error;
    }
}

async function generateWorkload() {
    const startTime = Date.now();
    let totalRequests = 0;
    let totalLatency = 0;
    let totalFirstTokenLatency = 0;
    let sloAttainmentCount = 0;

    // Generate requests
    const requests = generateRequests({
        numAdapters: n,
        alpha: alpha,
        reqRate: R,
        cv: cv,
        duration: traceDuration / 1000,
        inputRange: [Il, Iu],
        outputRange: [Ol, Ou],
        seed: 42
    });

    // Initialize progress bar
    const bar = new ProgressBar('Processing requests [:bar] :current/:total (:percent) :etas', {
        complete: '=',
        incomplete: ' ',
        width: 40,
        total: Math.floor(R * (traceDuration / 1000))
    });

    // Process requests sequentially
    for (const req of requests) {
        // Calculate intended start time
        const intendedStartTime = startTime + req.time;
        
        // Wait until intended start time
        const waitTime = intendedStartTime - Date.now();
        if (waitTime > 0) {
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }

        try {
            const prompt = "Hello ".repeat(req.inputLength).trim();
            const requestStartTime = performance.now();

            await setLoraAdapterScales(req.adapter_id, n);
            
            const response = await fetch("http://127.0.0.1:8080/completion", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    n_predict: req.outputLength,
                })
            });

            const result = await response.json();
            const requestEndTime = performance.now();
            const requestLatency = requestEndTime - intendedStartTime;
            const firstTokenLatency = requestStartTime - intendedStartTime + result.first_token_latency;
            // print this three values
            // console.log(`requestStartTime: ${requestStartTime}, intendedStartTime: ${intendedStartTime}, firstTokenLatency: ${result.firstTokenLatency}`);

            totalRequests++;
            totalLatency += requestLatency / 1000; // Convert to seconds
            totalFirstTokenLatency += firstTokenLatency;
            if (firstTokenLatency <= 6000) {
                sloAttainmentCount++;
            }

            bar.tick();

        } catch (error) {
            console.error(`Request failed for adapter ${req.adapter_ids}:`, error);
            bar.tick();
        }
    }
    
    // Print statistics
    const elapsedTime = (Date.now() - startTime) / 1000;
    console.log(`\nTotal requests: ${totalRequests}`);
    console.log(`Average latency: ${(elapsedTime / totalRequests).toFixed(2)} s`);
    console.log(`Average first token latency: ${(totalFirstTokenLatency / totalRequests).toFixed(2)} ms`);
    console.log(`Throughput: ${(totalRequests / elapsedTime).toFixed(2)} req/s`);
    console.log(`SLO attainment: ${((sloAttainmentCount / totalRequests) * 100).toFixed(2)}%`);
}

generateWorkload();
