const prompt = `Building a website can be done in 10 simple steps:`;

async function Test() {
    let response = await fetch("http://127.0.0.1:8080/completion", {
        method: 'POST',
        body: JSON.stringify({
            prompt,
            n_predict: 8,
            adapter_idx: 2,
        })
    })
    console.log((await response.json()).content)
}

Test()