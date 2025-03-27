const gamma = require('gamma'); // Library to generate Gamma distributed random numbers
const ProgressBar = require('progress');

// Configuration parameters
const n = 20; // Number of adapters
const alpha = 1; // Power-law exponent
const R = 0.5; // Total request rate in requests per second
const cv = 1; // Coefficient of variance
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

    const totalRequests = Math.floor(reqRate * duration);
    
    // Generate adapter IDs using power law distribution (3 per request)
    const generateAdapterId = () => {
        const prob = Math.random();
        const powerLaw = Math.pow(prob, 1/alpha);
        return Math.floor(powerLaw * numAdapters);
    };
    
    const adapterIds = Array(totalRequests).fill(0).map(() => {
        return [
            generateAdapterId(),
            generateAdapterId(),
            generateAdapterId()
        ];
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
            adapter_ids: adapterIds[i], // Now an array of 3 adapter IDs
            inputLength: inputLengths[i],
            outputLength: outputLengths[i]
        });
    }

    return requests;
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
        duration: traceDuration / 1000, // Convert to seconds
        inputRange: [Il, Iu],
        outputRange: [Ol, Ou],
        seed: 42
    });

    // Earlier in the code, before sending requests:
    const bar = new ProgressBar('Processing requests [:bar] :current/:total (:percent) :etas', {
        complete: '=',
        incomplete: ' ',
        width: 40,
        total:  Math.floor(R * (traceDuration / 1000))
    });

    // Send requests with proper timing
    const sendRequests = requests.map(async (req) => {
        const waitTime = req.time - (Date.now() - startTime);
        if (waitTime > 0) {
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }

        try {
            prompt = "Hello ".repeat(req.inputLength).trim()
            const requestStartTime = performance.now();
            const response = await fetch("http://127.0.0.1:8080/completion", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    n_predict: req.outputLength,
                    adapter_ids: req.adapter_ids
                })
            });

            const result = await response.json();

            // Validate result structure
            if (!result || typeof result !== 'object') {
                throw new Error('Invalid response format');
            }

            if (!('first_token_latency' in result)) {
                throw new Error('Missing first_token_latency in response');
            }

            const requestEndTime = performance.now() + result.prompt_process_time;
            const requestLatency = requestEndTime - requestStartTime  + result.prompt_process_time;
            const firstTokenLatency = result.first_token_latency + result.prompt_process_time;

            totalRequests++;
            totalLatency += requestLatency;
            totalFirstTokenLatency += firstTokenLatency;
            if (firstTokenLatency <= 6000) {
                sloAttainmentCount++;
            }

            completedRequests++;
            bar.tick();

        } catch (error) {
            console.error(`Request failed for adapter ${req.adapter_id}:`, error);
        }
    });

    await Promise.all(sendRequests);
    
    // Print statistics
    const elapsedTime = (Date.now() - startTime) / 1000;
    console.log(`\nTotal requests: ${totalRequests}`);
    console.log(`Average latency: ${(totalLatency / totalRequests).toFixed(2)} ms`);
    console.log(`Average first token latency: ${(totalFirstTokenLatency / totalRequests).toFixed(2)} ms`);
    console.log(`Throughput: ${(totalRequests / elapsedTime).toFixed(2)} req/s`);
    console.log(`SLO attainment: ${((sloAttainmentCount / totalRequests) * 100).toFixed(2)}%`);
}

generateWorkload();