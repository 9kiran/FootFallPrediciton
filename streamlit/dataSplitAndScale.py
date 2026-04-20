from sklearn.preprocessing import StandardScaler
import streamlit as st

def dataSplitAndScale(X, Y):
        # X and Y are already in chronological order

    # Calculate the split point (80% point)
    split_idx = int(len(X) * 0.8)

    st.write(f"Total records: {len(X)}")
    st.write(f"Training set: {split_idx} records")
    st.write(f"Test set: {len(X) - split_idx} records")

    # Split manually to preserve time order
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = Y.iloc[:split_idx]
    y_test = Y.iloc[split_idx:]

    st.write(f"\nTraining data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    st.write(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")


    # 1. Initialize the Scaler
    sc = StandardScaler()

    # 2. FIT and TRANSFORM the training data
    # This learns the mean/std from X_train and scales it
    X_train_scaled = sc.fit_transform(X_train)

    # 3. TRANSFORM the test data
    # We DO NOT 'fit' on test data; we use the training parameters
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
