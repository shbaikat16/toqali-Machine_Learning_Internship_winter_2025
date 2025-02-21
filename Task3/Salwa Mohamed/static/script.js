document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    let reservationDate = document.getElementById("reservation_date").value.split("-");
    
    let inputData = {
        "Booking_ID": document.getElementById("Booking_ID").value,
        "number_of_adults": document.getElementById("number_of_adults").value,
        "number_of_children": document.getElementById("number_of_children").value,
        "number_of_weekend_nights": document.getElementById("number_of_weekend_nights").value,
        "number_of_week_nights": document.getElementById("number_of_week_nights").value,
        "type_of_meal": document.getElementById("type_of_meal").value,
        "room_type": document.getElementById("room_type").value,
        "lead_time": document.getElementById("lead_time").value,
        "market_segment_type": document.getElementById("market_segment_type").value,
        "P_C": document.getElementById("P_C").value,
        "P_not_C": document.getElementById("P_not_C").value,
        "average_price": document.getElementById("average_price").value,
        "special_requests": document.getElementById("special_requests").value,
        "reservation_day": reservationDate[2],
        "reservation_month": reservationDate[1],
        "reservation_year": reservationDate[0]
    };

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(inputData)
    })
    .then(response => response.json())
    .then(data => document.getElementById("result").innerText = "Booking Status: " + data.prediction);
});
