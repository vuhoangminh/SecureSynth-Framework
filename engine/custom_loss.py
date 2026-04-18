import torch
import time


class CorrelationLoss(torch.nn.Module):
    # https://stackoverflow.com/a/19710598/11170350
    def __init__(self):
        super(CorrelationLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))
        # Normalise X and Y
        X = X - X.mean(1)[:, None]
        Y = Y - Y.mean(1)[:, None]
        # Standardise X and Y
        X = X / (X.std(1)[:, None] + self.epsilon)
        Y = Y / (Y.std(1)[:, None] + self.epsilon)
        # multiply X and Y
        Z = (X * Y).mean(1)
        Z = 1 - Z.mean()
        return Z


class CorrectedCorrelationLossNotEfficient(torch.nn.Module):
    def __init__(self):
        super(CorrectedCorrelationLossNotEfficient, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_correlation_each_pair(self, x, y):
        """
        Calculates the correlation coefficient between two PyTorch tensors.

        Args:
            self (CorrelationLoss): An instance of the CorrelationLoss class. (Optional for some use cases)
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor with the same shape as x.

        Returns:
            torch.Tensor: The correlation coefficient between x and y.
        """
        x = (x - x.mean()) / (x.std() + self.epsilon)
        y = (y - y.mean()) / (y.std() + self.epsilon)
        z = (x * y).mean()
        return z

    def compute_loss(self, X, Y):
        """
        Iterates through each pair of columns in a PyTorch tensor and applies function f.

        Args:
            X (torch.Tensor): The first input tensor of shape (batch_size, num_features).
            Y (torch.Tensor): The second input tensor of the same shape as X.

        Returns:
            torch.Tensor: The calculated correlation loss.
        """

        loss = 0
        count = 0
        num_columns = X.shape[1]
        for i in range(num_columns):
            for j in range(i + 1, num_columns):
                p = self.compute_correlation_each_pair(
                    X[:, i], X[:, j]
                ) - self.compute_correlation_each_pair(Y[:, i], Y[:, j])
                loss += p**2
                count += 1
        return loss / count

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = self.compute_loss(X, Y)
        return loss


class CorrectedCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(CorrectedCorrelationLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_loss(self, X, Y):
        """
        Calculates the correlation loss between X and Y.

        Args:
            X (torch.Tensor): The first input tensor of shape (batch_size, num_features).
            Y (torch.Tensor): The second input tensor of the same shape as X.

        Returns:
            torch.Tensor: The calculated correlation loss.
        """
        X_normalized = (X - X.mean(dim=0)) / (X.std(dim=0) + self.epsilon)
        Y_normalized = (Y - Y.mean(dim=0)) / (Y.std(dim=0) + self.epsilon)

        # Compute the cross-correlation matrix
        X_cross_corr_matrix = torch.mm(X_normalized.T, X_normalized) / X.shape[0]
        Y_cross_corr_matrix = torch.mm(Y_normalized.T, Y_normalized) / Y.shape[0]

        # Exclude diagonal elements (self-correlations)
        X_cross_corr_matrix -= torch.diag(X_cross_corr_matrix.diag())
        Y_cross_corr_matrix -= torch.diag(Y_cross_corr_matrix.diag())

        # Calculate the squared difference between cross-correlations
        diff_matrix = X_cross_corr_matrix - Y_cross_corr_matrix
        loss = (diff_matrix**2).sum() / (X.shape[1] * (X.shape[1] - 1))

        return loss

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = self.compute_loss(X, Y)
        return loss


class DWPLoss(torch.nn.Module):
    def __init__(self):
        super(DWPLoss, self).__init__()
        return

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))
        # Normalise X and Y
        X = torch.sum(X, 0)
        Y = torch.sum(Y, 0)
        Z = torch.sum(torch.abs(X / torch.sum(X) - Y / torch.sum(Y)))
        return Z


class NewDistributionLoss1(torch.nn.Module):
    def __init__(self):
        super(NewDistributionLoss1, self).__init__()
        return

    def compute_moments_along_rows(self, data):
        """
        Calculates the mean, variance, skewness, and kurtosis of a 2D PyTorch tensor along rows.

        Args:
            data (torch.Tensor): The 2D PyTorch tensor for which to compute the moments.

        Returns:
            tuple: A tuple containing tensors of mean, variance, skewness, and kurtosis along rows.
        """

        # Check if the input is a 2D tensor
        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D tensor.")

        # Reduce mean along rows
        mean = torch.mean(data, dim=0)

        # Reduce std along rows
        std = torch.std(data, dim=0)

        # Reduce variance along rows (unbiased)
        variance = torch.var(data, dim=0)

        # Centering data for moment calculations (avoiding in-place operations)
        centered_data = data - mean

        # Reduce skewness along rows (unbiased)
        skewness = torch.mean((centered_data / std) ** 3, dim=0)

        # Reduce kurtosis along rows (unbiased)
        kurtosis = torch.mean((centered_data / std) ** 4, dim=0) - 3

        return mean, variance, skewness, kurtosis

    def forward(
        self,
        X,
        Y,
        p_mean=1,
        p_variance=1,
        p_skewness=1,
        p_kurtosis=1,
    ):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        mean_X, variance_X, skewness_X, kurtosis_X = self.compute_moments_along_rows(X)
        mean_Y, variance_Y, skewness_Y, kurtosis_Y = self.compute_moments_along_rows(Y)

        Z = p_mean * torch.mean(torch.abs(mean_X - mean_Y))
        Z += p_variance * torch.mean(torch.abs(variance_X - variance_Y))
        Z += p_skewness * torch.mean(torch.abs(skewness_X - skewness_Y))
        Z += p_kurtosis * torch.mean(torch.abs(kurtosis_X - kurtosis_Y))

        return Z


class NewDistributionLoss2(torch.nn.Module):
    def __init__(self):
        super(NewDistributionLoss2, self).__init__()
        return

    def calculate_S(self, x, n, t):
        """
        Calculates the value of S based on the equation:

        S = 1/N * sum(i=1 to N) * sum(d=0 to n) * x_i^d * exp(t * x_i)

        Args:
            x (torch.Tensor): A 1D PyTorch tensor representing the x_i values.
            n (int): The upper limit of the summation over d.
            t (float): The parameter t.

        Returns:
            torch.Tensor: The calculated value of S.
        """

        N = len(
            x
        )  # Get the number of elements in the tensor (assuming they represent x_i)
        devices = x.device  # Get the device of the tensor (CPU or GPU)

        # Create tensors for 1 and exponent term (avoid repeated calculations)
        one = torch.tensor(1, dtype=x.dtype, device=devices)
        exp_term = torch.exp(t * x)

        S = 0
        S += (x ** (n + 1) - one) / (x - one) * exp_term

        # Sum over elements and normalize by N
        S = torch.sum(S) / N

        return S

    def forward(self, X, Y):
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        n_cols = X.shape[1]

        loss = 0
        n, t = 4, 1
        n, t = 4, 0
        # n, t = 10, 0
        # n, t = 4, -1
        # n, t = 4, -2
        for j in range(n_cols):
            loss += (
                self.calculate_S(X[:, j], n, t) - self.calculate_S(Y[:, j], n, t)
            ) ** 2

        return loss / n_cols


class DistributionLoss(torch.nn.Module):
    """
    This class implements a loss function that compares the distribution moments
    (mean, variance, higher-order moments) between two tensors.

    Args:
        None: This class does not require any arguments during initialization.

    Returns:
        torch.Tensor: The total loss calculated based on the differences in distribution moments.
    """

    def __init__(self):
        super(DistributionLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_moments_along_rows(self, data, rank, dim=0):
        """
        Calculates the specified moment (mean, variance, higher-order moments)
        along the specified dimension (default: 0) of a 2D tensor.

        Args:
            data (torch.Tensor): The input tensor (must be 2D).
            rank (int): The order of the moment to calculate (e.g., 1 for mean, 2 for variance).
            dim (int, optional): The dimension along which to reduce (default: 0 for rows).

        Raises:
            ValueError: If the input data is not a 2D tensor.

        Returns:
            torch.Tensor: The calculated moment along the specified dimension.
        """

        if len(data.shape) > 2:
            raise ValueError("Input data must be a 1D or 2D tensors.")

        # Reduce moment along rows
        if rank == 1:
            return torch.mean(data, dim=dim)
        elif rank == 2:
            return torch.var(data, dim=dim)
        else:
            centered_data = data - torch.mean(
                data, dim=dim
            )  # Centering for higher moments
            return torch.mean(
                (centered_data / (torch.std(data, dim=dim) + self.epsilon)) ** rank,
                dim=dim,
            )  # add small value to avoid division by 0

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float = 1.0,
        n: int = 4,
    ) -> torch.Tensor:
        """
        Calculates the distribution loss between two tensors X and Y.

        Args:
            X (torch.Tensor): The real tensor.
            Y (torch.Tensor): The synthetic tensor.
            alha/lambda (float, optional): Weighting factor for the loss terms (default: 1.0).
            n (int, optional): The number of moments to compare (default: 4).

        Raises:
            AssertionError: If any element in X or Y is NaN.

        Returns:
            torch.Tensor: The total loss calculated based on the moment differences.
        """

        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = 0
        for rank in range(1, n + 1):  # Start from 1 to exclude mean (rank 0)
            moment_real = self.compute_moments_along_rows(X, rank)
            moment_syn = self.compute_moments_along_rows(Y, rank)
            l = (
                1 - (moment_syn + self.epsilon) / (moment_real + self.epsilon)
            ) ** 2  # add small value to avoid division by 0
            loss += alpha / rank * l

        return loss.mean()


class NormalizedDistributionLoss(torch.nn.Module):
    """
    This class implements a loss function that compares the distribution moments
    (mean, variance, higher-order moments) between two tensors.

    Args:
        None: This class does not require any arguments during initialization.

    Returns:
        torch.Tensor: The total loss calculated based on the differences in distribution moments.
    """

    def __init__(self):
        super(NormalizedDistributionLoss, self).__init__()
        self.epsilon = 1e-5
        return

    def compute_moments_along_rows(self, data, rank, dim=0):
        """
        Calculates the specified moment (mean, variance, higher-order moments)
        along the specified dimension (default: 0) of a 2D tensor.

        Args:
            data (torch.Tensor): The input tensor (must be 2D).
            rank (int): The order of the moment to calculate (e.g., 1 for mean, 2 for variance).
            dim (int, optional): The dimension along which to reduce (default: 0 for rows).

        Raises:
            ValueError: If the input data is not a 2D tensor.

        Returns:
            torch.Tensor: The calculated moment along the specified dimension.
        """

        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D tensor.")

        # Reduce moment along rows
        if rank == 1:
            return torch.mean(data, dim=dim)
        elif rank == 2:
            return torch.var(data, dim=dim)
        else:
            centered_data = data - torch.mean(
                data, dim=dim
            )  # Centering for higher moments
            return torch.mean(
                (centered_data / (torch.std(data, dim=dim) + self.epsilon)) ** rank,
                dim=dim,
            )  # add small value to avoid division by 0

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float = 1.0,
        n: int = 4,
    ) -> torch.Tensor:
        """
        Calculates the distribution loss between two tensors X and Y.

        Args:
            X (torch.Tensor): The real tensor.
            Y (torch.Tensor): The synthetic tensor.
            alpha (float, optional): Weighting factor for the loss terms (default: 1.0).
            n (int, optional): The number of moments to compare (default: 4).

        Raises:
            AssertionError: If any element in X or Y is NaN.

        Returns:
            torch.Tensor: The total loss calculated based on the moment differences.
        """

        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = 0
        for rank in range(1, n + 1):  # Start from 1 to exclude mean (rank 0)
            moment_real = self.compute_moments_along_rows(X, rank)
            moment_syn = self.compute_moments_along_rows(Y, rank)
            # l = (
            #     moment_syn / torch.sum(moment_syn)
            #     - moment_real / torch.sum(moment_real)
            # ) ** 2  # too small
            l = torch.abs(
                moment_syn / torch.sum(moment_syn)
                - moment_real / torch.sum(moment_real)
            )

            loss += alpha / rank * l
        return loss.mean()


class DistributionLossCTAB(torch.nn.Module):
    """
    This class implements a loss function that compares the distribution moments
    (mean, variance, higher-order moments) between two tensors.

    Args:
        None: This class does not require any arguments during initialization.

    Returns:
        torch.Tensor: The total loss calculated based on the differences in distribution moments.
    """

    def __init__(self):
        super(DistributionLossCTAB, self).__init__()
        return

    def compute_moments_along_rows(self, data, rank, dim=0):
        """
        Calculates the specified moment (mean, variance, higher-order moments)
        along the specified dimension (default: 0) of a 2D tensor.

        Args:
            data (torch.Tensor): The input tensor (must be 2D).
            rank (int): The order of the moment to calculate (e.g., 1 for mean, 2 for variance).
            dim (int, optional): The dimension along which to reduce (default: 0 for rows).

        Raises:
            ValueError: If the input data is not a 2D tensor.

        Returns:
            torch.Tensor: The calculated moment along the specified dimension.
        """

        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D tensor.")

        # Reduce moment along rows
        if rank == 1:
            return torch.mean(data, dim=dim)
        elif rank == 2:
            return torch.var(data, dim=dim)
        else:
            centered_data = data - torch.mean(
                data, dim=dim
            )  # Centering for higher moments
            return torch.mean(
                (centered_data / torch.std(data, dim=dim)) ** rank, dim=dim
            )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        alpha: float = 1.0,
        n: int = 4,
    ) -> torch.Tensor:
        """
        Calculates the distribution loss between two tensors X and Y.

        Args:
            X (torch.Tensor): The real tensor.
            Y (torch.Tensor): The synthetic tensor.
            alpha (float, optional): Weighting factor for the loss terms (default: 1.0).
            n (int, optional): The number of moments to compare (default: 4).

        Raises:
            AssertionError: If any element in X or Y is NaN.

        Returns:
            torch.Tensor: The total loss calculated based on the moment differences.
        """

        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))

        loss = 0
        for rank in range(1, n + 1):  # Start from 1 to exclude mean (rank 0)
            moment_real = self.compute_moments_along_rows(X, rank)
            moment_syn = self.compute_moments_along_rows(Y, rank)
            l = (
                moment_real - moment_syn
            ) ** 2 / moment_real.sum() ** 2  # my suggestion
            # l = (1 - moment_syn / moment_real) ** 2  # Tommy's suggestion
            """No, the size of the dataset should not cause invalid values.
            NaN values in parameters are most likely caused by NaNs in the gradient, 
            which might be caused by an exploding loss or e.g. invalid input values.
            """
            loss += alpha / rank * l
        return loss.mean()


def test_loss_function(loss_fn, input1, input2):
    """
    Tests the custom loss function and checks requires_grad for the first input.

    Args:
        loss_fn: Your custom loss function (callable).
        input1 (torch.Tensor): The first input to the loss function.
        input2 (torch.Tensor): The second input to the loss function.
    """

    start = time.time()

    # Set requires_grad to True for input1 (if not already set)
    if not input1.requires_grad:
        input1.requires_grad = True

    # Calculate the loss
    output = loss_fn(input1, input2)

    # Print the loss value
    print(f"Loss: {output.item()}")

    # Check requires_grad for input1
    print(f"Input 1 requires_grad: {input1.requires_grad}")
    print(f"Output requires_grad: {output.requires_grad}")

    end = time.time()
    print(f"Elapsed time is:", end - start)


def main():
    # Example usage (assuming your loss function is named 'my_loss_function')
    ncols = 100
    data1 = (
        torch.randn(40000, ncols, requires_grad=True) * 1
    )  # Ensure requires_grad for input1
    data2 = torch.randn(25000, ncols) * 2

    # test_loss_function(DistributionLoss(), data1, data2)
    # test_loss_function(NewCorrelationLoss(), data1, data2)
    # test_loss_function(NewDistributionLoss2(), data1, data2)
    # test_loss_function(DistributionLoss(), data1, data2)

    # test_loss_function(CorrectedCorrelationLossNotEfficient(), data1, data2)
    # test_loss_function(CorrectedCorrelationLoss(), data1, data2)
    test_loss_function(NormalizedDistributionLoss(), data1, data2)


if __name__ == "__main__":
    main()
