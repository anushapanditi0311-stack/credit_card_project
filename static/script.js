document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';

    // Pack input form values into a JSON payload
    const payload = {
        gender: document.getElementById('gender').value,
        own_car: document.getElementById('own_car').value,
        own_realty: document.getElementById('own_realty').value,
        children: document.getElementById('children').value,
        income: document.getElementById('income').value,
        income_type: document.getElementById('income_type').value,
        education: document.getElementById('education').value,
        family_status: document.getElementById('family_status').value,
        housing_type: document.getElementById('housing_type').value,
        occupation: document.getElementById('occupation').value,
        family_size: document.getElementById('family_size').value,
        age: document.getElementById('age').value,
        years_employed: document.getElementById('years_employed').value,
        unemployed: document.getElementById('unemployed').value
    };

    try {
        // Send POST request containing payload to the prediction endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        
        resultDiv.style.display = 'block';
        if (result.approved) {
            resultDiv.className = 'success';
            resultDiv.innerText = 'Application Status: APPROVED';
        } else {
            resultDiv.className = 'danger';
            resultDiv.innerText = 'Application Status: REJECTED';
        }
    } catch (err) {
        resultDiv.className = 'danger';
        resultDiv.innerText = 'Error processing credit application.';
        console.error(err);
    }
});
