# Importing dependencies from the functions file
from functions import *

# Running the code
if __name__ == "__main__":
    # Generate the model
    model = generate_model()
    print("Model architecture generated!")

    # Checking if the model weights are saved
    if os.path.exists(weights_filename):
        # Load the model weights
        model.load_state_dict(torch.load(weights_filename))
        print("Model weights loaded")

    # Specifying the device type
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backend.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Fitting model to device
    model.to(device)

    # Hyperparameters
    p1_epochs = 5
    p2_epochs = 15
    batch_size = 8
    regularization = 0.001
    margin = 1
    lr = 0.0001
    workers = 4
    offset = 0.2

    # Defining the optimizer for both phases
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler1 = StepLR(optimizer, step_size=1, gamma=0.1)

    # Defining the preprocessing transformations
    transform = transforms.Compose([transforms.Lambda(prepare_input)])

    # Defining an array of positive (or deepfake) and negative (or real) images
    fake_train = [new_train_dir + x for x in os.listdir(new_train_dir) if "fake" in x]
    real_train = [new_train_dir + x for x in os.listdir(new_train_dir) if "real" in x]
    fake_test = [new_test_dir + x for x in os.listdir(new_test_dir) if "fake" in x]
    real_test = [new_test_dir + x for x in os.listdir(new_test_dir) if "real" in x]
    fake_val = [new_val_dir + x for x in os.listdir(new_val_dir) if "fake" in x]
    real_val = [new_val_dir + x for x in os.listdir(new_val_dir) if "real" in x]

    # Generating the datasets for phase 1 using our custom class
    # Offset avoids data repetition between phase 1 and 2
    train1_dataset = Phase1Data(
        new_train_dir,
        transform=transform,
        fakes=fake_train[: int(len(fake_train) * offset)],
        reals=real_train[: int(len(real_train) * offset)],
        offset=offset,
    )
    test1_dataset = Phase1Data(
        new_test_dir,
        transform=transform,
        fakes=fake_test[: int(len(fake_test) * offset)],
        reals=real_test[: int(len(real_test) * offset)],
        offset=offset,
    )
    val1_dataset = Phase1Data(
        new_val_dir,
        transform=transform,
        fakes=fake_val[: int(len(fake_val) * offset)],
        reals=real_val[: int(len(real_val) * offset)],
        offset=offset,
    )

    # Generating the dataloaders for phase 1
    train1_loader = DataLoader(
        train1_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    test1_loader = DataLoader(
        test1_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val1_loader = DataLoader(
        val1_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # Clear the cache
    torch.cuda.empty_cache()

    print("data loaders generated!")

    # Phase 1 training
    phase1_train(
        model,
        train1_loader,
        optimizer,
        margin,
        device,
        p1_epochs,
        scheduler1,
        val1_loader,
    )

    # Phase 1 testing
    phase1_test(model, test1_loader, margin, device)

    # Scheduler
    scheduler2 = StepLR(optimizer, step_size=1, gamma=0.1)

    # Cross entropy loss function for phase 2
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Generating the datasets for phase 2 using our custom class
    train2_dataset = Phase2Data(new_train_dir, transform=transform, offset=offset)
    test2_dataset = Phase2Data(new_test_dir, transform=transform, offset=offset)
    val2_dataset = Phase2Data(new_val_dir, transform=transform, offset=offset)

    # Generating the dataloaders for phase 2
    train2_loader = DataLoader(
        train2_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    test2_loader = DataLoader(
        test2_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val2_loader = DataLoader(
        val2_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )

    # Phase 2 training
    phase2_train(
        model,
        train2_loader,
        optimizer,
        cross_entropy_loss,
        device,
        p2_epochs,
        regularization,
        scheduler2,
        val2_loader,
    )

    # Phase 2 testing
    phase2_test(model, test2_loader, device)
