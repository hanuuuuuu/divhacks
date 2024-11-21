document.addEventListener('DOMContentLoaded', () => {
    // Add event listener for the submit button
    const submitButton = document.querySelector('button[type="button"]');
    submitButton.addEventListener('click', submitForm);
});

async function submitForm() {
    const form = document.getElementById('crisisForm');
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseInt(value);
    });

    try {
        console.log('Submitting form data:', data); // Log the data being sent

        const response = await fetch('http://127.0.0.1:5501/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Server response:', result); // Log the server response
        alert(`Crisis Risk: ${result.crisis_risk}`);
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
        alert('An error occurred while submitting the form. Please try again.');
    }
}
