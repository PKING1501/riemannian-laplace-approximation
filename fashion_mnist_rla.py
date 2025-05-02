"""
File containing the FashionMNIST experiments with Riemannian Laplace Approximation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from laplace import Laplace
import seaborn as sns
from torch import nn
from manifold import cross_entropy_manifold, linearized_cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from utils.metrics import accuracy, nll, brier, calibration
from sklearn.metrics import brier_score_loss
import geomai.utils.geometry as geometry
import argparse
from torchmetrics.functional.classification import calibration_error
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# FashionMNIST class names
fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def main(args):
    # Initialize settings
    palette = sns.color_palette("colorblind")
    print(f"Linearization: {args.linearized_pred}")
    print(f"Optimizing prior: {args.optimize_prior}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Seed: {args.seed}")

    # Load FashionMNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST specific mean/std
    ])
    
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    # Visualize: One image per class from the original dataset
    print("Displaying one example per class from FashionMNIST...")
    examples_per_class = {}
    for img, label in train_dataset:
        if label not in examples_per_class:
            examples_per_class[label] = img
        if len(examples_per_class) == 10:
            break

    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for cls, ax in enumerate(axes):
        ax.imshow(examples_per_class[cls].squeeze(), cmap="gray")
        ax.set_title(fashion_classes[cls], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.suptitle("FashionMNIST Examples", y=1.1)
    plt.show()

    # Split into train and validation
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model architecture - slightly larger for FashionMNIST
    num_features = 28 * 28
    num_output = 10
    H = 512  # Increased hidden size for more complex dataset

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, H),
        nn.ReLU(),  # Changed to ReLU for better performance
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, num_output)
    ).to(device)

    # Training setup - longer training for FashionMNIST
    if args.optimizer == "sgd":
        weight_decay = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        max_epoch = 30  # Increased epochs
    else:
        weight_decay = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        max_epoch = 20  # Increased epochs

    loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    # Training loop with progress bar
    best_valid_acc = 0
    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{max_epoch}") as tepoch:
            for batch_img, batch_label in tepoch:
                batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_img)
                loss = loss_criterion(outputs, batch_label)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        valid_accuracy = 0
        with torch.no_grad():
            for batch_img, batch_label in valid_loader:
                batch_img, batch_label = batch_img.to(device), batch_label.to(device)
                outputs = model(batch_img)
                preds = torch.argmax(outputs, dim=1)
                valid_accuracy += (preds == batch_label).sum().item()

        valid_accuracy /= len(valid_loader.dataset)
        train_loss /= len(train_loader.dataset)

        # Early stopping check
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), 'best_model.pt')

        print(f"Epoch {epoch+1}/{max_epoch}: Train Loss: {train_loss:.4f}, Valid Acc: {valid_accuracy:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Store MAP solution
    map_solution = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    torch.nn.utils.vector_to_parameters(map_solution, model.parameters())

    # Fit Laplace approximation
    print("Fitting Laplace Approximation")
    la = Laplace(
        model,
        "classification",
        subset_of_weights=args.subset,
        hessian_structure=args.structure,
        prior_precision=2 * weight_decay,
    )
    la.fit(train_loader)

    if args.optimize_prior:
        la.optimize_prior_precision(method="marglik")
    print(f"Prior precision: {la.prior_precision}")

    # Sample from Laplace posterior
    n_last_layer_weights = num_output * H + num_output if args.subset == "last_layer" else None
    
    if args.subset == "last_layer" and args.structure == "diag":
        samples = torch.randn(args.samples, la.n_params, device=device)
        samples *= la.posterior_scale.reshape(1, la.n_params)
        V_LA = samples.cpu().numpy()
    elif args.subset == "last_layer":
        dist = MultivariateNormal(
            loc=torch.zeros(n_last_layer_weights, device=device), 
            scale_tril=la.posterior_scale
        )
        V_LA = dist.sample((args.samples,)).cpu().numpy()
    elif args.structure == "diag":
        samples = torch.randn(args.samples, la.n_params, device=device)
        samples *= la.posterior_scale.reshape(1, la.n_params)
        V_LA = samples.cpu().numpy()
    else:
        dist = MultivariateNormal(
            loc=torch.zeros_like(map_solution), 
            scale_tril=la.posterior_scale
        )
        V_LA = dist.sample((args.samples,)).cpu().numpy()

    print(f"Sampled parameter matrix shape: {V_LA.shape}")

    # Create manifold for Riemannian approach
    if args.linearized_pred:
        with torch.no_grad():
            f_MAP = []
            y_train_all = []
            for batch_img, batch_label in train_loader:
                f_MAP.append(model(batch_img.to(device)))
                y_train_all.append(batch_label)
            f_MAP = torch.cat(f_MAP)
            y_train = torch.cat(y_train_all)

        if args.subset == "last_layer":
            feature_extractor_map = map_solution[:-n_last_layer_weights]
            ll_map = map_solution[-n_last_layer_weights:]
            
            feature_extractor_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU()
            ).to(device)
            ll = nn.Linear(H, num_output).to(device)
            
            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            with torch.no_grad():
                R = []
                for batch_img, _ in train_loader:
                    R.append(feature_extractor_model(batch_img.to(device)))
                R = torch.cat(R)

            manifold = linearized_cross_entropy_manifold(
                ll, R, y_train, f_MAP=f_MAP, theta_MAP=ll_map, 
                batching=False, lambda_reg=la.prior_precision.item()/2 if args.optimize_prior else weight_decay
            )
        else:
            model2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, num_output)
            ).to(device)
            
            manifold = linearized_cross_entropy_manifold(
                model2,
                train_loader if args.batch_data else (
                    torch.cat([x for x, _ in train_loader]),
                    torch.cat([y for _, y in train_loader])
                ),
                f_MAP=f_MAP,
                theta_MAP=map_solution,
                batching=args.batch_data,
                lambda_reg=la.prior_precision.item()/2 if args.optimize_prior else weight_decay
            )
    else:
        if args.subset == "last_layer":
            feature_extractor_map = map_solution[:-n_last_layer_weights]
            ll_map = map_solution[-n_last_layer_weights:]
            
            feature_extractor_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU()
            ).to(device)
            ll = nn.Linear(H, num_output).to(device)
            
            torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
            torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

            with torch.no_grad():
                R = []
                for batch_img, _ in train_loader:
                    R.append(feature_extractor_model(batch_img.to(device)))
                R = torch.cat(R)

            manifold = cross_entropy_manifold(
                ll, R, torch.cat([y for _, y in train_loader]), 
                batching=False, 
                lambda_reg=la.prior_precision.item()/2 if args.optimize_prior else weight_decay
            )
        else:
            model2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, num_output)
            ).to(device)
            
            manifold = cross_entropy_manifold(
                model2,
                train_loader if args.batch_data else (
                    torch.cat([x for x, _ in train_loader]),
                    torch.cat([y for _, y in train_loader])
                ),
                batching=args.batch_data,
                lambda_reg=la.prior_precision.item()/2 if args.optimize_prior else weight_decay
            )

    # Solve exponential map for Riemannian samples
    weights_ours = torch.zeros(args.samples, len(map_solution), device=device)
    for n in tqdm(range(args.samples), desc="Solving expmap"):
        v = torch.from_numpy(V_LA[n, :]).float().to(device).reshape(-1, 1)

        if args.subset == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            _new_ll_weights = torch.from_numpy(curve(1)[0]).float().to(device) if isinstance(curve(1)[0], np.ndarray) else curve(1)[0]
            _new_weights = torch.cat((feature_extractor_map.view(-1), _new_ll_weights.view(-1)))
            weights_ours[n, :] = _new_weights
        else:
            if args.expmap_different_batches:
                n_sub_data = 150
                idx_sub = np.random.choice(len(train_loader.dataset), n_sub_data, replace=False)
                sub_data = [train_loader.dataset[i] for i in idx_sub]
                sub_x = torch.stack([x for x, _ in sub_data]).to(device)
                sub_y = torch.stack([y for _, y in sub_data]).to(device)
                
                if args.linearized_pred:
                    sub_f_MAP = []
                    for x, _ in sub_data:
                        sub_f_MAP.append(model(x.unsqueeze(0).to(device)))
                    sub_f_MAP = torch.cat(sub_f_MAP)
                    
                    manifold = linearized_cross_entropy_manifold(
                        model2, sub_x, sub_y, f_MAP=sub_f_MAP, theta_MAP=map_solution,
                        batching=False, lambda_reg=la.prior_precision.item()/2,
                        N=len(train_loader.dataset), B1=n_sub_data
                    )
                else:
                    manifold = cross_entropy_manifold(
                        model2, sub_x, sub_y, batching=False,
                        lambda_reg=la.prior_precision.item()/2,
                        N=len(train_loader.dataset), B1=n_sub_data
                    )

            curve, failed = geometry.expmap(manifold, map_solution.clone(), v)
            weights_ours[n, :] = torch.from_numpy(curve(1)[0]).float().to(device) if isinstance(curve(1)[0], np.ndarray) else curve(1)[0]

    # Get Laplace weights
    weights_LA = torch.zeros(args.samples, len(map_solution), device=device)
    for n in range(args.samples):
        v_n = torch.from_numpy(V_LA[n, :]).float().to(device)
        if args.subset == "last_layer":
            laplace_weights = torch.cat((
                feature_extractor_map.clone().view(-1),
                (ll_map + v_n).view(-1)
            ))
        else:
            laplace_weights = map_solution + v_n
        weights_LA[n, :] = laplace_weights

    # Evaluate on test set
    def evaluate(weights, model, loader):
        model.eval()
        probs = []
        labels = []
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(weights, model.parameters())
            for batch_img, batch_label in loader:
                batch_img = batch_img.to(device)
                outputs = model(batch_img)
                probs.append(torch.softmax(outputs, dim=1))
                labels.append(batch_label)
        return torch.cat(probs), torch.cat(labels)

    # MAP evaluation
    torch.nn.utils.vector_to_parameters(map_solution, model.parameters())
    P_test_MAP, y_test = evaluate(map_solution, model, test_loader)

    # Riemannian evaluation
    P_test_OURS = 0
    for n in range(args.samples):
        P_test_OURS += evaluate(weights_ours[n], model, test_loader)[0]
    P_test_OURS /= args.samples

    # Laplace evaluation
    P_test_LAPLACE = 0
    for n in range(args.samples):
        P_test_LAPLACE += evaluate(weights_LA[n], model, test_loader)[0]
    P_test_LAPLACE /= args.samples

    # Compute metrics
    def compute_metrics(probs, labels, name):
        acc = accuracy(probs, labels)
        nll_val = nll(probs, labels)
        brier_val = brier(probs, labels)
        ece = calibration_error(probs, labels, norm="l1", task="multiclass", num_classes=10, n_bins=10) * 100
        mce = calibration_error(probs, labels, norm="max", task="multiclass", num_classes=10, n_bins=10) * 100
        
        print(f"\nResults {name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  NLL: {nll_val:.4f}")
        print(f"  Brier: {brier_val:.4f}")
        print(f"  ECE: {ece:.4f}")
        print(f"  MCE: {mce:.4f}")
        
        return {
            "Accuracy": acc,
            "NLL": nll_val,
            "Brier": brier_val,
            "ECE": ece,
            "MCE": mce
        }

    dict_MAP = compute_metrics(P_test_MAP, y_test, "MAP")
    dict_LA = compute_metrics(P_test_LAPLACE, y_test, "Laplace")
    dict_OUR = compute_metrics(P_test_OURS, y_test, "Riemannian")

    # Visualization: Confidence histogram
    plt.figure(figsize=(10, 5))
    plt.hist(P_test_MAP.max(1)[0].cpu().numpy(), bins=20, alpha=0.5, label='MAP')
    plt.hist(P_test_LAPLACE.max(1)[0].cpu().numpy(), bins=20, alpha=0.5, label='Laplace')
    plt.hist(P_test_OURS.max(1)[0].cpu().numpy(), bins=20, alpha=0.5, label='Riemannian')
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("FashionMNIST Confidence Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualization: t-SNE of features
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    features = []
    labels = []
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            features.append(feature_extractor(img).cpu())
            labels.append(label)
    
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title("FashionMNIST t-SNE Visualization of Test Features")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Class-wise accuracy comparison
    def class_wise_accuracy(probs, labels):
        class_correct = torch.zeros(10)
        class_total = torch.zeros(10)
        preds = torch.argmax(probs, dim=1)
        for i in range(10):
            idx = (labels == i)
            class_correct[i] = (preds[idx] == labels[idx]).sum().item()
            class_total[i] = idx.sum().item()
        return class_correct / class_total

    acc_map = class_wise_accuracy(P_test_MAP, y_test)
    acc_la = class_wise_accuracy(P_test_LAPLACE, y_test)
    acc_our = class_wise_accuracy(P_test_OURS, y_test)

    plt.figure(figsize=(12, 6))
    x = np.arange(10)
    width = 0.25
    plt.bar(x - width, acc_map, width, label='MAP')
    plt.bar(x, acc_la, width, label='Laplace')
    plt.bar(x + width, acc_our, width, label='Riemannian')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy Comparison')
    plt.xticks(x, fashion_classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "results_MAP": dict_MAP,
        "results_Laplace": dict_LA,
        "results_Riemannian": dict_OUR
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riemannian Laplace Approximation on FashionMNIST")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam", help="Optimizer")
    parser.add_argument("--optimize_prior", action="store_true", help="Optimize prior precision")
    parser.add_argument("--batch_data", action="store_true", help="Use batch data for manifold")
    parser.add_argument("--structure", choices=["diag", "full"], default="diag", help="Hessian structure")
    parser.add_argument("--subset", choices=["last_layer", "all"], default="all", help="Subset of weights")
    parser.add_argument("--samples", type=int, default=20, help="Number of posterior samples")
    parser.add_argument("--linearized_pred", action="store_true", help="Use linearized predictions")
    parser.add_argument("--expmap_different_batches", action="store_true", help="Use different batches for expmap")
    
    args = parser.parse_args()
    main(args)