from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Serve the HTML frontend
@app.route("/")
def index():
    with open("index.html", "r") as f:
        html_content = f.read()
    return render_template_string(html_content)

@app.route("/run-model", methods=["POST"])
def run_model():
    params = request.get_json()
    df = pd.read_csv("escooter_synthetic_data.csv")

    # Parameters
    mass = params["mass"]
    g = 9.81
    crr = params["crr"]
    rho = params["rho"]
    cd = params["cd"]
    A = params["A"]
    motor_eff = params["motor_efficiency"]
    brake_eff = params["brake_efficiency"]

    df["F_roll"] = mass * g * crr
    df["F_drag"] = 0.5 * rho * cd * A * df["Speed (m/s)"]**2
    df["F_slope"] = mass * g * df["Elevation (m)"].diff().fillna(0)
    df["F_accel"] = mass * df["Acceleration (m/s^2)"]
    df["F_total"] = df["F_roll"] + df["F_drag"] + df["F_slope"] + df["F_accel"]
    df["Power_W"] = (df["F_total"] * df["Speed (m/s)"]).clip(lower=0)
    df["Energy_Wh"] = df["Power_W"] / 3600
    df["Regen_Wh"] = 0
    df.loc[df["Brake Applied"], "Regen_Wh"] = (
        -df["F_accel"] * df["Speed (m/s)"] / 3600 * brake_eff
    ).clip(lower=0)
    df["Heat_Loss_W"] = df["Power_W"] * (1 - motor_eff)
    df["Motor_Temp_Est"] = 25 + df["Heat_Loss_W"].cumsum() * 0.0002

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Time (s)"], df["Speed (m/s)"], label="Speed (m/s)")
    ax.plot(df["Time (s)"], df["Energy_Wh"].cumsum(), label="Energy Used (Wh)")
    ax.plot(df["Time (s)"], df["Motor_Temp_Est"], label="Motor Temp (°C)")
    ax.set_title("E-Scooter Ride Summary")
    ax.legend()
    ax.grid()

    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({"plot": image_base64})

if __name__ == "__main__":
    app.run(debug=True)
