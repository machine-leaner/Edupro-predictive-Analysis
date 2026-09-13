"""
EduPro Predictive Analytics Dashboard
--------------------------------------
Streamlit app for course demand and revenue analytics/prediction.

Run with:  streamlit run app.py

Loads pre-computed processed data and pre-trained models (see src/) rather
than retraining on every page load.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

st.set_page_config(page_title="EduPro Predictive Analytics", layout="wide", page_icon="\U0001F4CA")


# ----------------------------------------------------------------------
# Cached data / model loaders
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    course_df = pd.read_csv(os.path.join(DATA_DIR, "course_level_dataset.csv"),
                             parse_dates=["FirstTransactionDate", "LastTransactionDate"])
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions_processed.csv"),
                                parse_dates=["TransactionDate"])
    users = pd.read_csv(os.path.join(DATA_DIR, "users_processed.csv"))
    teachers = pd.read_csv(os.path.join(DATA_DIR, "teachers_processed.csv"))
    forecast_panel = pd.read_csv(os.path.join(DATA_DIR, "forecasting_panel.csv"))
    return course_df, transactions, users, teachers, forecast_panel


@st.cache_resource
def load_models():
    demand_model = joblib.load(os.path.join(MODELS_DIR, "demand_model.pkl"))
    revenue_model = joblib.load(os.path.join(MODELS_DIR, "revenue_model.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return demand_model, revenue_model, meta


@st.cache_data
def load_reports():
    reports = {}
    for name in ["enrollment_model_comparison", "revenue_model_comparison",
                 "forecast_model_comparison", "enrollment_feature_importance",
                 "revenue_feature_importance"]:
        path = os.path.join(REPORTS_DIR, f"{name}.csv")
        if os.path.exists(path):
            reports[name] = pd.read_csv(path)
    return reports


course_df, transactions, users, teachers, forecast_panel = load_data()
demand_model, revenue_model, meta = load_models()
reports = load_reports()

st.sidebar.title("\U0001F4DA EduPro Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Dashboard",
        "Dataset Overview",
        "Course Demand Analysis",
        "Revenue Analysis",
        "Demand Prediction",
        "Revenue Prediction",
        "Category Analysis",
        "Feature Importance",
        "Business Recommendations",
    ],
)

CATEGORY_OPTIONS = sorted(course_df["CourseCategory"].unique())
TYPE_OPTIONS = sorted(course_df["CourseType"].unique())
LEVEL_OPTIONS = sorted(course_df["CourseLevel"].unique())


# ----------------------------------------------------------------------
# 1. Executive Dashboard
# ----------------------------------------------------------------------
if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    st.caption("EduPro Online Platform \u2014 course demand & revenue at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Courses", f"{course_df['CourseID'].nunique():,}")
    c2.metric("Total Teachers", f"{teachers['TeacherID'].nunique():,}")
    c3.metric("Total Users", f"{users['UserID'].nunique():,}")
    c4.metric("Total Transactions", f"{len(transactions):,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Total Enrollments", f"{course_df['EnrollmentCount'].sum():,}")
    c6.metric("Total Revenue", f"${course_df['TotalRevenue'].sum():,.2f}")
    c7.metric("Avg Course Rating", f"{course_df['CourseRating'].mean():.2f} / 5")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        trans_m = transactions.copy()
        trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
        monthly = trans_m.groupby("Month").size().reset_index(name="Enrollments")
        fig = px.line(monthly, x="Month", y="Enrollments", markers=True,
                      title="Monthly Enrollment Trend")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        monthly_rev = trans_m.groupby("Month")["Amount"].sum().reset_index(name="Revenue")
        fig = px.line(monthly_rev, x="Month", y="Revenue", markers=True,
                      title="Monthly Revenue Trend", color_discrete_sequence=["#C44E52"])
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cat_enroll = course_df.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values().reset_index()
        fig = px.bar(cat_enroll, x="EnrollmentCount", y="CourseCategory", orientation="h",
                     title="Enrollment by Category")
        st.plotly_chart(fig, use_container_width=True)
    with col_d:
        cat_rev = course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values().reset_index()
        fig = px.bar(cat_rev, x="TotalRevenue", y="CourseCategory", orientation="h",
                     title="Revenue by Category", color_discrete_sequence=["#C44E52"])
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# 2. Dataset Overview
# ----------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.subheader("Source Tables")
    t1, t2, t3, t4 = st.tabs(["Users", "Teachers", "Courses", "Transactions"])
    with t1:
        st.write(f"Shape: {users.shape}")
        st.dataframe(users.head(20), use_container_width=True)
    with t2:
        st.write(f"Shape: {teachers.shape}")
        st.dataframe(teachers.head(20), use_container_width=True)
    with t3:
        st.write(f"Shape: {course_df[['CourseID','CourseName','CourseCategory','CourseType','CourseLevel','CoursePrice','CourseDuration','CourseRating']].shape}")
        st.dataframe(course_df[['CourseID','CourseName','CourseCategory','CourseType','CourseLevel','CoursePrice','CourseDuration','CourseRating']], use_container_width=True)
    with t4:
        st.write(f"Shape: {transactions.shape}")
        st.dataframe(transactions.head(20), use_container_width=True)

    st.divider()
    st.subheader("Data Quality Summary")
    st.markdown("""
    - **No missing values and no duplicate rows** were found in any of the four sheets.
    - **All foreign keys resolve**: every UserID, CourseID, and TeacherID referenced in
      Transactions exists in its parent sheet.
    - **Course \u2194 Teacher relationship is many-to-many**, present only inside Transactions
      (Courses/Teachers have no direct link). Teacher features used in modeling are
      therefore historical aggregates over the instructor pool per course, not a fixed join.
    - **Transaction Amount always exactly equals the course's CoursePrice** (0 exceptions) \u2014
      there is no variation from discounts/refunds in this dataset, so **TotalRevenue is a
      deterministic function of CoursePrice \u00d7 EnrollmentCount.**
    - 38 of 60 courses are free (CoursePrice = 0), which shows up as 6,403 of 10,000
      Amount = 0 transactions.
    """)

    st.subheader("Final Course-Level Modeling Dataset")
    st.write(f"Shape: {course_df.shape}")
    st.dataframe(course_df.describe(include="all").transpose(), use_container_width=True)


# ----------------------------------------------------------------------
# 3. Course Demand Analysis
# ----------------------------------------------------------------------
elif page == "Course Demand Analysis":
    st.title("Course Demand Analysis")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        sel_cat = f1.multiselect("Category", CATEGORY_OPTIONS, default=CATEGORY_OPTIONS)
        sel_type = f2.multiselect("Course Type", TYPE_OPTIONS, default=TYPE_OPTIONS)
        sel_level = f3.multiselect("Course Level", LEVEL_OPTIONS, default=LEVEL_OPTIONS)
        date_range = st.date_input(
            "Transaction date range",
            value=(transactions["TransactionDate"].min().date(), transactions["TransactionDate"].max().date()),
        )

    filtered = course_df[
        course_df["CourseCategory"].isin(sel_cat)
        & course_df["CourseType"].isin(sel_type)
        & course_df["CourseLevel"].isin(sel_level)
    ]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        trans_filtered = transactions[
            (transactions["TransactionDate"] >= start) & (transactions["TransactionDate"] <= end)
            & transactions["CourseID"].isin(filtered["CourseID"])
        ]
    else:
        trans_filtered = transactions[transactions["CourseID"].isin(filtered["CourseID"])]

    c1, c2 = st.columns(2)
    with c1:
        cat_enroll = filtered.groupby("CourseCategory")["EnrollmentCount"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(cat_enroll, x="EnrollmentCount", y="CourseCategory", orientation="h",
                                title="Enrollment by Category"), use_container_width=True)
    with c2:
        top10 = filtered.nlargest(10, "EnrollmentCount")[["CourseName", "EnrollmentCount"]]
        st.plotly_chart(px.bar(top10.sort_values("EnrollmentCount"), x="EnrollmentCount", y="CourseName",
                                orientation="h", title="Top 10 Courses by Enrollment"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        level_demand = filtered.groupby("CourseLevel")["EnrollmentCount"].sum().reset_index()
        st.plotly_chart(px.bar(level_demand, x="CourseLevel", y="EnrollmentCount",
                                title="Demand by Course Level"), use_container_width=True)
    with c4:
        type_demand = filtered.groupby("CourseType")["EnrollmentCount"].sum().reset_index()
        st.plotly_chart(px.bar(type_demand, x="CourseType", y="EnrollmentCount",
                                title="Demand by Course Type", color_discrete_sequence=["#55A868"]),
                         use_container_width=True)

    trans_m = trans_filtered.copy()
    trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
    monthly = trans_m.groupby("Month").size().reset_index(name="Enrollments")
    st.plotly_chart(px.line(monthly, x="Month", y="Enrollments", markers=True,
                             title="Monthly Enrollment Trend (filtered)"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.scatter(filtered, x="CoursePrice", y="EnrollmentCount", color="CourseType",
                                    hover_data=["CourseName"], title="Price vs Enrollment"),
                         use_container_width=True)
    with c6:
        st.plotly_chart(px.scatter(filtered, x="CourseRating", y="EnrollmentCount", color="CourseLevel",
                                    hover_data=["CourseName"], title="Rating vs Enrollment"),
                         use_container_width=True)

    st.caption("Correlation between Course Rating and Enrollment across all 60 courses is "
               "weakly positive (~0.29); Price shows a weak negative correlation (~-0.16). "
               "These are associations only, not causal claims.")


# ----------------------------------------------------------------------
# 4. Revenue Analysis
# ----------------------------------------------------------------------
elif page == "Revenue Analysis":
    st.title("Revenue Analysis")

    st.info("Amount in this dataset always exactly equals CoursePrice, so TotalRevenue = "
            "CoursePrice \u00d7 EnrollmentCount deterministically \u2014 revenue trends here mirror "
            "price and enrollment patterns rather than an independent signal.")

    c1, c2 = st.columns(2)
    c1.metric("Total Revenue", f"${course_df['TotalRevenue'].sum():,.2f}")
    c2.metric("Avg Revenue per Course", f"${course_df['TotalRevenue'].mean():,.2f}")

    c3, c4 = st.columns(2)
    with c3:
        cat_rev = course_df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values().reset_index()
        st.plotly_chart(px.bar(cat_rev, x="TotalRevenue", y="CourseCategory", orientation="h",
                                title="Revenue by Category"), use_container_width=True)
    with c4:
        top10_rev = course_df.nlargest(10, "TotalRevenue")[["CourseName", "TotalRevenue"]]
        st.plotly_chart(px.bar(top10_rev.sort_values("TotalRevenue"), x="TotalRevenue", y="CourseName",
                                orientation="h", title="Top 10 Courses by Revenue",
                                color_discrete_sequence=["#C44E52"]), use_container_width=True)

    trans_m = transactions.copy()
    trans_m["Month"] = trans_m["TransactionDate"].dt.to_period("M").astype(str)
    monthly_rev = trans_m.groupby("Month")["Amount"].sum().reset_index(name="Revenue")
    st.plotly_chart(px.line(monthly_rev, x="Month", y="Revenue", markers=True,
                             title="Monthly Revenue Trend", color_discrete_sequence=["#C44E52"]),
                     use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(px.scatter(course_df, x="CoursePrice", y="TotalRevenue", color="CourseCategory",
                                    hover_data=["CourseName"], title="Revenue vs Price"),
                         use_container_width=True)
    with c6:
        st.plotly_chart(px.scatter(course_df, x="EnrollmentCount", y="TotalRevenue", color="CourseCategory",
                                    hover_data=["CourseName"], title="Revenue vs Enrollment"),
                         use_container_width=True)


# ----------------------------------------------------------------------
# 5. Demand Prediction
# ----------------------------------------------------------------------
elif page == "Demand Prediction":
    st.title("Demand Prediction")
    st.caption(f"Model: {meta['enrollment_model']} (selected by lowest LOOCV MAE)")

    st.warning("This model's out-of-sample R\u00b2 is near zero (Leave-One-Out cross-validation "
               "on 60 courses) \u2014 course/teacher attributes in this dataset show very little "
               "predictive signal for enrollment. Treat predictions as a rough, low-confidence "
               "estimate, not a reliable forecast.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Course Category", CATEGORY_OPTIONS)
        course_type = st.selectbox("Course Type", TYPE_OPTIONS)
        level = st.selectbox("Course Level", LEVEL_OPTIONS)
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=1000.0, value=100.0, step=10.0)
    with col2:
        duration = st.number_input("Course Duration (hours)", min_value=0.5, max_value=100.0, value=15.0, step=0.5)
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, 0.1)
        teacher_experience = st.slider("Avg Teacher Experience (years)", 1, 25, 8)
        teacher_rating = st.slider("Avg Teacher Rating", 1.0, 5.0, 4.0, 0.1)
    distinct_teachers = st.slider("Distinct Teachers Expected", 1, 30, 10)

    if st.button("Predict Enrollment", type="primary"):
        inputs = {
            "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
            "AvgTeacherRating": teacher_rating, "AvgTeacherExperience": teacher_experience,
            "DistinctTeacherCount": distinct_teachers,
            "CourseCategory": category, "CourseType": course_type, "CourseLevel": level,
        }
        cols = meta["enrollment_features_numeric"] + meta["enrollment_features_categorical"]
        row = pd.DataFrame([{c: inputs[c] for c in cols}])
        pred = demand_model.predict(row)[0]
        st.success(f"### Predicted Enrollment: {pred:,.0f} students")

        st.markdown("**Major factors influencing enrollment predictions in this model** "
                    "(from RandomForest feature importance on the same data): CourseRating and "
                    "CourseDuration rank highest, followed by the size/experience of the "
                    "instructor pool. These are model associations, not proven causes.")


# ----------------------------------------------------------------------
# 6. Revenue Prediction
# ----------------------------------------------------------------------
elif page == "Revenue Prediction":
    st.title("Revenue Prediction")
    st.caption(f"Model: {meta['revenue_model']} (selected by lowest LOOCV MAE)")
    st.info("Predictions are estimates based on historical patterns, not guarantees.")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Course Category", CATEGORY_OPTIONS, key="rev_cat")
        course_type = st.selectbox("Course Type", TYPE_OPTIONS, key="rev_type")
        level = st.selectbox("Course Level", LEVEL_OPTIONS, key="rev_level")
        price = st.number_input("Course Price ($)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0, key="rev_price")
    with col2:
        duration = st.number_input("Course Duration (hours)", min_value=0.5, max_value=100.0, value=15.0, step=0.5, key="rev_dur")
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, 0.1, key="rev_rating")
        teacher_experience = st.slider("Avg Teacher Experience (years)", 1, 25, 8, key="rev_exp")
        teacher_rating = st.slider("Avg Teacher Rating", 1.0, 5.0, 4.0, 0.1, key="rev_teacher_rating")
    distinct_teachers = st.slider("Distinct Teachers Expected", 1, 30, 10, key="rev_distinct")

    if st.button("Predict Revenue", type="primary"):
        inputs = {
            "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
            "AvgTeacherRating": teacher_rating, "AvgTeacherExperience": teacher_experience,
            "DistinctTeacherCount": distinct_teachers,
            "CourseCategory": category, "CourseType": course_type, "CourseLevel": level,
        }
        cols = meta["revenue_features_numeric"] + meta["revenue_features_categorical"]
        row = pd.DataFrame([{c: inputs[c] for c in cols}])
        pred_rev = revenue_model.predict(row)[0]

        # Also show implied demand model prediction for context.
        demand_cols = meta["enrollment_features_numeric"] + meta["enrollment_features_categorical"]
        demand_row = pd.DataFrame([{c: inputs[c] for c in demand_cols}])
        pred_enroll = demand_model.predict(demand_row)[0]

        st.success(f"### Predicted Revenue (Estimate): ${pred_rev:,.2f}")
        c1, c2 = st.columns(2)
        c1.metric("Category", category)
        c2.metric("Expected Demand (from Demand model)", f"{pred_enroll:,.0f} students")
        st.caption("Because Amount always equals CoursePrice in this dataset, revenue is "
                    "almost entirely driven by the price you set \u00d7 predicted enrollment \u2014 "
                    "CoursePrice alone explains ~97% of the RandomForest's revenue importance.")


# ----------------------------------------------------------------------
# 7. Category Analysis
# ----------------------------------------------------------------------
elif page == "Category Analysis":
    st.title("Category Analysis")

    cat_summary = course_df.groupby("CourseCategory").agg(
        Courses=("CourseID", "count"),
        TotalEnrollment=("EnrollmentCount", "sum"),
        TotalRevenue=("TotalRevenue", "sum"),
        AvgPrice=("CoursePrice", "mean"),
        AvgRating=("CourseRating", "mean"),
    ).sort_values("TotalRevenue", ascending=False).reset_index()

    st.dataframe(cat_summary.style.format({
        "TotalRevenue": "${:,.2f}", "AvgPrice": "${:,.2f}", "AvgRating": "{:.2f}"
    }), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(cat_summary, x="CourseCategory", y="TotalEnrollment",
                                title="Total Enrollment by Category"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(cat_summary, x="CourseCategory", y="TotalRevenue",
                                title="Total Revenue by Category", color_discrete_sequence=["#C44E52"]),
                         use_container_width=True)

    trans_cat = transactions.merge(course_df[["CourseID", "CourseCategory"]], on="CourseID", how="left")
    trans_cat["Month"] = trans_cat["TransactionDate"].dt.to_period("M").astype(str)
    top5 = cat_summary.nlargest(5, "TotalRevenue")["CourseCategory"].tolist()
    cat_month = trans_cat[trans_cat["CourseCategory"].isin(top5)].groupby(
        ["Month", "CourseCategory"])["Amount"].sum().reset_index()
    st.plotly_chart(px.line(cat_month, x="Month", y="Amount", color="CourseCategory",
                             title="Monthly Revenue Trend \u2014 Top 5 Categories"), use_container_width=True)


# ----------------------------------------------------------------------
# 8. Feature Importance
# ----------------------------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance / Explainability")
    st.caption("RandomForest impurity-based importance \u2014 reflects predictive association "
               "within the model, not proof of causation.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enrollment (Demand) Prediction")
        if "enrollment_feature_importance" in reports:
            imp = reports["enrollment_feature_importance"].head(12).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"),
                             use_container_width=True)
        st.caption("Note: the enrollment model's LOOCV R\u00b2 is near zero, so these rankings "
                   "should be read as weak, exploratory signals only.")
    with col2:
        st.subheader("Revenue Prediction")
        if "revenue_feature_importance" in reports:
            imp = reports["revenue_feature_importance"].head(12).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                                    color_discrete_sequence=["#C44E52"]), use_container_width=True)
        st.caption("CoursePrice dominates revenue importance (~97%) because Amount always "
                   "equals CoursePrice in this dataset.")

    st.divider()
    st.subheader("Model Comparison Tables (LOOCV, n=60 courses)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Enrollment models**")
        if "enrollment_model_comparison" in reports:
            st.dataframe(reports["enrollment_model_comparison"], use_container_width=True)
    with c2:
        st.markdown("**Revenue models**")
        if "revenue_model_comparison" in reports:
            st.dataframe(reports["revenue_model_comparison"], use_container_width=True)

    st.markdown("**Forecasting models (chronological split, next-month enrollment)**")
    if "forecast_model_comparison" in reports:
        st.dataframe(reports["forecast_model_comparison"], use_container_width=True)


# ----------------------------------------------------------------------
# 9. Business Recommendations
# ----------------------------------------------------------------------
elif page == "Business Recommendations":
    st.title("Business Recommendations")

    top_enroll_cat = course_df.groupby("CourseCategory")["EnrollmentCount"].sum().idxmax()
    top_rev_cat = course_df.groupby("CourseCategory")["TotalRevenue"].sum().idxmax()
    price_corr = course_df["CoursePrice"].corr(course_df["EnrollmentCount"])
    rating_corr = course_df["CourseRating"].corr(course_df["EnrollmentCount"])

    st.markdown(f"""
    **1. Strongest observed demand:** *{top_enroll_cat}* leads all categories in total
    historical enrollment.

    **2. Strongest observed revenue:** *{top_rev_cat}* generates the highest total revenue,
    driven primarily by its pricing rather than a demand advantage (recall Amount =
    CoursePrice \u00d7 EnrollmentCount deterministically in this dataset).

    **3. Price vs. demand:** the correlation between CoursePrice and EnrollmentCount across
    all 60 courses is **{price_corr:.2f}** \u2014 a weak negative association. Price alone is not
    a strong predictor of course popularity here.

    **4. Rating vs. demand:** the correlation between CourseRating and EnrollmentCount is
    **{rating_corr:.2f}** \u2014 a weak positive association, the strongest single numeric
    association found for demand, but still not strong enough to build a confident
    forecasting model (LOOCV R\u00b2 \u2248 0).

    **5. Teacher effects:** average instructor rating/experience per course show negligible
    correlation with enrollment in this dataset (see Feature Importance page).

    **6. Most influential model features:** CourseRating and CourseDuration rank highest for
    enrollment; CoursePrice overwhelmingly dominates revenue (a mechanical consequence of the
    data, not a business insight about pricing elasticity).

    **7 & 8. Categories to watch:** {top_enroll_cat} (demand) and {top_rev_cat} (revenue) are
    the categories with the strongest historical track record and are reasonable starting
    points for new course investment, though the weak overall model signal means this should
    be paired with qualitative market research before committing budget.

    **9. Seasonality:** monthly enrollment volume across 2025 ranges from about 762 to 899
    transactions/month \u2014 mild month-to-month variation with no dramatic seasonal spike.

    **10. Pricing:** because revenue is a deterministic function of price, EduPro can compute
    exact revenue impact of a price change once expected enrollment is estimated \u2014 but this
    dataset does not show strong evidence that lower prices meaningfully increase enrollment
    (price/enrollment correlation \u2248 {price_corr:.2f}), so blanket discounting is not
    obviously supported by the data.
    """)

    st.warning("**Honesty note:** the enrollment (demand) models in this project show "
               "near-zero out-of-sample explanatory power (LOOCV R\u00b2 \u2248 0 for all algorithms "
               "tried). This is a genuine finding, not a modeling failure to hide \u2014 with only "
               "60 courses and largely price-independent enrollment counts, this dataset does "
               "not contain a strong learnable demand signal from catalog attributes alone. "
               "Recommendations above should be read as descriptive observations from the "
               "historical data, not as validated causal drivers of demand.")
