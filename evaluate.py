import torch

def RMSE(model, criterion, test_dataloader, device):
    total_loss = 0
    with torch.no_grad():
        for destination, time, sex, age, dayofweek, month, day, congestion_1 in test_dataloader:
            # itemId
            destination = destination.to(device)
            # user information(userId)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device), age.to(device), month.to(device), day.to(device)
            # rating(target)
            congestion_1 = congestion_1.to(device)

            prediction = model(dayofweek, time, sex, age, month, day, destination)
            loss = criterion(prediction, congestion_1)
            loss = torch.sqrt(loss)

            total_loss += loss
    return total_loss/len(test_dataloader)

def RMSE_con(model, criterion, test_dataloader, device):
    total_loss = 0
    with torch.no_grad():
        for destination, time, dayofweek, month, day, congestion in test_dataloader:
            # itemId
            destination = destination.to(device)
            # user information(userId)
            dayofweek, time, month, day = dayofweek.to(device), time.to(device), month.to(device), day.to(device)
            # rating(target)
            congestion = congestion.to(device)

            prediction = model(dayofweek, time, month, day, destination)
            loss = criterion(prediction, congestion)
            loss = torch.sqrt(loss)

            total_loss += loss
    return total_loss/len(test_dataloader)