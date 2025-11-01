import os
import sys
from django.conf import settings
from django.core.management import execute_from_command_line
from django.shortcuts import render
from django.http import HttpResponse
from django.urls import path

# === Import ML/LLM models ===
from model.knn import knn_predict
from model.linear_regression import train_and_predict
from model.llm_interface import generate_training_program


# === Django minimal setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="hackabot_secret",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
        TEMPLATES=[{
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [
                os.path.join(BASE_DIR, "bot/templates"),
                os.path.join(BASE_DIR, "templates"),
            ],
            "APP_DIRS": True,
        }],
        STATIC_URL="/static/",
        STATICFILES_DIRS=[
            os.path.join(BASE_DIR, "bot/static"),
            os.path.join(BASE_DIR, "static"),
        ],
    )


# === Views ===

def index(request):
    """Homepage — shows buttons to navigate to other features"""
    return render(request, "index.html")


def generate_form(request):
    """Render the user input page for generating plans"""
    return render(request, "generate.html")


def generate_plan(request):
    """Generate a personalized training plan"""
    if request.method != "POST":
        return HttpResponse("Invalid request", status=400)

    # Extract data from POST
    age = request.POST.get("age")
    height_cm = request.POST.get("height_cm")
    weight_kg = request.POST.get("weight_kg")
    gender = request.POST.get("gender")
    sport = request.POST.get("sport")
    duration_weeks = int(request.POST.get("duration_weeks") or 4)
    fitness_level = request.POST.get("fitness_level")
    equipment = request.POST.get("equipment")
    injuries = request.POST.get("injuries") or "none"

    # Collect goals
    goals_list = []
    if request.POST.get("goal_strength"): goals_list.append("strength")
    if request.POST.get("goal_endurance"): goals_list.append("endurance")
    if request.POST.get("goal_speed"): goals_list.append("speed/agility")
    if request.POST.get("goal_power"): goals_list.append("power")
    if request.POST.get("goal_weight_loss"): goals_list.append("weight_loss")

    goals = ", ".join(goals_list) if goals_list else "general fitness"

    # Generate plan
    plan = generate_training_program(
        sport=sport,
        goal=goals,
        duration_weeks=duration_weeks,
        fitness_level=fitness_level
    )

    # Pass to template
    context = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "gender": gender,
        "sport": sport,
        "duration_weeks": duration_weeks,
        "fitness_level": fitness_level,
        "equipment": equipment,
        "injuries": injuries,
        "goals": goals,
        "plan": plan,
    }
    return render(request, "bot/result.html", context)


# === Other ML endpoints (optional) ===

def predict_knn(request):
    if request.method == "POST":
        price = float(request.POST.get("price"))
        common = float(request.POST.get("common"))
        dataset = "data/car_knn_dataset.csv"

        brand = knn_predict(
            a=[price, common],
            path=dataset,
            x_col="price_kUSD",
            y_col="common",
            label_col="brand",
            k=4
        )
        return HttpResponse(f"<h1>Predicted Brand: {brand}</h1><a href='/'>Go Back</a>")
    return HttpResponse("Invalid request", status=400)


def train_linear(request):
    if request.method == "POST":
        dataset = "data/linear_regression_dataset.csv"
        x_col = request.POST.get("x_col")
        y_col = request.POST.get("y_col")
        L = float(request.POST.get("learning_rate"))
        epochs = int(request.POST.get("epochs"))
        m, b = train_and_predict(dataset, x_col, y_col, L, epochs)
        return HttpResponse(f"<h1>Trained Model: y = {m:.3f}x + {b:.3f}</h1><a href='/'>Go Back</a>")
    return HttpResponse("Invalid request", status=400)


# === URL routes ===
urlpatterns = [
    path("", index, name="home"),
    path("generate/", generate_form, name="generate_form"),
    path("generate_plan/", generate_plan, name="generate_plan"),
    path("predict_knn/", predict_knn, name="predict_knn"),
    path("train_linear/", train_linear, name="train_linear"),
]


# === Run the app ===
if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "__main__")
    execute_from_command_line(sys.argv)

