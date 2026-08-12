
// Set today's date automatically
const dateInput = document.getElementById("date");

if (dateInput) {
    const today = new Date();

    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, "0");
    const day = String(today.getDate()).padStart(2, "0");

    dateInput.value = `${year}-${month}-${day}`;
}


// Confirm before deleting an expense
const deleteForms = document.querySelectorAll(".delete-form");

deleteForms.forEach((form) => {
    form.addEventListener("submit", (event) => {

        const confirmed = confirm(
            "Are you sure you want to delete this expense?"
        );

        if (!confirmed) {
            event.preventDefault();
        }
    });
});
