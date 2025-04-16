export default async function GenerateImages() {
  try {
    const inputField = document.getElementById("inputField");
    if (!inputField) {
      console.error("Input field not found");
      return;
    }

    const input = inputField.value;
    const response = await fetch("http://localhost:5000/run-test5", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: input }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const resultElement = document.getElementById("show-result");
    if (resultElement) {
      resultElement.innerText = "GENERATED";
    }
  } catch (error) {
    console.error("Error:", error);
    const resultElement = document.getElementById("show-result");
    if (resultElement) {
      resultElement.innerText = "Error occurred. Check console.";
    }
  }
}
