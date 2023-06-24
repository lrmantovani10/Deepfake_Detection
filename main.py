# Importing dependencies from the functions file
from functions import *

# Running the code
if __name__ == "__main__":
    # Checking if the model is saved
    if os.path.exists(model_filename):
        # Load the model
        model = pk.load(open(model_filename, "rb"))
        print("Model loaded")
    else:
        # Generate the model
        model = generate_model()
        # Save the model
        # pk.dump(model, open("model.pkl", "wb"))
        print("Model saved")

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
    batch_size = 1
    regularization = 0.001
    margin = 0.8
    lr = 0.0001
    workers = 2
    offset = 0.2

    # Defining the optimizer for both phases
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler1 = StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler2 = StepLR(optimizer, step_size=1, gamma=0.1)

    # Cross entropy loss function for phase 2
    cross_entropy_loss = nn.CrossEntropyLoss()

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

    # Generating the datasets for phase 2 using our custom class
    train2_dataset = Phase2Data(new_train_dir, transform=transform, offset=offset)
    test2_dataset = Phase2Data(new_test_dir, transform=transform, offset=offset)
    val2_dataset = Phase2Data(new_val_dir, transform=transform, offset=offset)

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

    print("data loaders generated!")

    # Phase 1 training
    phase1_train(
        model,
        train1_loader,
        optimizer,
        margin,
        device,
        p1_epochs
    )

    # Phase 1 validation
    phase1_val(
        model,
        val1_loader,
        optimizer,
        margin,
        device,
        scheduler1
    )

    # Phase 1 testing
    phase1_test(model, test1_loader, margin, device)

    # phase2()

# Determine if scheduler steps
# gate_cross = False
# # Phase 2 training
# for epoch in range(p2_epochs):

#     # Training phase
#     train(model, cross_entropy_loss, optimizer, scheduler, epoch, batch_size, regularization, device, new_train_dir)
#     # Testing phase
#     test(model, cross_entropy_loss, epoch, batch_size, regularization, device, new_test_dir


#     # Applying regularization to the cross entropy loss function
#     regularization_term = 0
#     for param in model.parameters():
#         # Adding L2 Norm to running sum
#         regularization_term += torch.norm(param)

#     # Multiplying by regularization factor and adding to cross entropy loss
#     cross_entropy_loss += regularization * regularization_term

#     # Backpropagation
#     loss.backward()


#### MOVE THIS NEXT PART TO VALIDATION (NOT TRAINING) STEP. Look at phase 1 implemnentation
#     if train2_acc >= 0.9 and epoch > 15:

#         if not gate_cross:
#             # Update the learning rate according to the scheduler
#             scheduler2.step()
#             scheduler2 = StepLR(optimizer, step_size=8, gamma=0.1)
#             gate_cross = True
#         else:
#             # Update the learning rate according to the scheduler
#             scheduler2.step()
