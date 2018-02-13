#ifndef CITYSCAPE_INFO_H_
#define CITYSCAPE_INFO_H_

namespace cityscape {

// This is the cityscape mapping to the defined labels of interest.
static int GetTrainId(int value) {
  int train_id = 255;
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      train_id = 255;
      break;
    case 7:
      train_id = 0;
      break;
    case 8:
      train_id = 1;
      break;
    case 9:
    case 10:
      train_id = 255;
      break;
    case 11:
      train_id = 2;
      break;
    case 12:
      train_id = 3;
      break;
    case 13:
      train_id = 4;
      break;
    case 14:
    case 15:
    case 16:
      train_id = 255;
      break;
    case 17:
      train_id = 5;
      break;
    case 18:
      train_id = 255;
      break;
    case 19:
      train_id = 6;
      break;
    case 20:
      train_id = 7;
      break;
    case 21:
      train_id = 8;
      break;
    case 22:
      train_id = 9;
      break;
    case 23:
      train_id = 10;
      break;
    case 24:
      train_id = 11;
      break;
    case 25:
      train_id = 12;
      break;
    case 26:
      train_id = 13;
      break;
    case 27:
      train_id = 14;
      break;
    case 28:
      train_id = 15;
      break;
    case 29:
    case 30:
      train_id = 255;
      break;
    case 31:
      train_id = 16;
      break;
    case 32:
      train_id = 17;
      break;
    case 33:
      train_id = 18;
      break;
    default:
      train_id = 255;
  }

  return train_id;
}

static int GetTrainIdFullUnsup(int value) {
  int train_id = 1;
  if (value == 0) train_id = 0;
  return train_id;
}

// Here are the cityscape ids that move.
static int GetTargetObjectIds(int value) {
  int id = 0;
  switch (value) {
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
      id = 1;
      break;
    default:
      id = 0;
  }
  return id;
}

// These are the most common cityscape ids that move
static int GetTargetSubsetObjectIds(int value) {
  int id = 0;
  switch (value) {
    case 11:
    case 13:
    case 18:
      id = 1;
      break;
    default:
      id = 0;
  }
  return id;
}

// This is the multiplier per class for cityscape to even out class imbalance.
static int GetMult(int train_id) {
  int mult = 1;
  switch (train_id) {
    case 11:
      mult = 2;
      break;
    case 18:
      mult = 7;
      break;
  }
  return mult;
}

}  // namespace cityscape

#endif  // CITYSCAPE_INFO_H_
