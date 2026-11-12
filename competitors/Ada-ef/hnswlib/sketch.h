#pragma once

#include <vector>
#include <memory>
#include <algorithm>

namespace hnswdis
{
    class Sketch
    {
    private:
        const std::vector<std::pair<int, std::vector<std::pair<int, float>>>> &ef_recall_estimators; // score, {(ef, recall)}, sorted by score
        const float expected_recall;
        std::vector<int> links;

    public:
        Sketch(
            const std::vector<std::pair<int, std::vector<std::pair<int, float>>>> &ef_recall_estimators,
            const float expected_recall) : ef_recall_estimators(ef_recall_estimators), expected_recall(expected_recall)
        {
            links.resize(101);
            int index = 0;
            for (int i = 0; i <= 100; ++i)
            {
                // Find the first element that is not smaller than `i` (b)
                while (index < ef_recall_estimators.size() && ef_recall_estimators[index].first < i)
                {
                    ++index;
                }

                // Determine `a` and `b`
                int a_index = (index > 0) ? index - 1 : -1;                       // Last smaller or equal element
                int b_index = (index < ef_recall_estimators.size()) ? index : -1; // First larger or equal element

                // Choose the closest one to `i`
                if (a_index != -1 && b_index != -1)
                {
                    // Both `a` and `b` exist, compare distances
                    if (std::abs(ef_recall_estimators[a_index].first - i) <= std::abs(ef_recall_estimators[b_index].first - i))
                    {
                        links[i] = a_index;
                    }
                    else
                    {
                        links[i] = b_index;
                    }
                }
                else if (a_index != -1)
                {
                    // Only `a` exists
                    links[i] = a_index;
                }
                else if (b_index != -1)
                {
                    // Only `b` exists
                    links[i] = b_index;
                }
            }
        }

        size_t get_entry(float score)
        {
            auto entry = std::lower_bound(ef_recall_estimators.begin(), ef_recall_estimators.end(), score, [](const auto &a, const float &b)
                                          { return a.first < b; });

            if (entry == ef_recall_estimators.begin())
            {
                entry = ef_recall_estimators.begin();
            }
            else if (entry == ef_recall_estimators.end())
            {
                entry = std::prev(ef_recall_estimators.end());
            }

            for (const auto &ef_recall : entry->second)
            {
                if (ef_recall.second >= expected_recall)
                {
                    return ef_recall.first;
                }
            }

            return entry->second.back().first;
        }

        size_t estimate_ef(float score) // deprecated
        {
            int index = links[(int)(score)];
            const auto &ef_recalls = ef_recall_estimators[index].second;
            for (const auto &ef_recall : ef_recalls)
            {
                if (ef_recall.second >= expected_recall)
                {
                    return ef_recall.first;
                }
            }
            return ef_recalls.back().first;
        }

        size_t estimate_ef2(float score)
        {
            size_t first = estimate_ef(score);

            if (score < 1 || score >= 100)
            {
                return first;
            }
        
            size_t second = estimate_ef(score - 1);
            size_t third = estimate_ef(score + 1);

            //Compute the median of the three values
            // return first + second + third - std::max({first, second, third}) - std::min({first, second, third});

            // Compute the average of the three values
            return (first + second + third) / 3;
        }

        void print()
        {
            std::cout << "Links: ";
            for (const auto &link : links)
            {
                std::cout << link << " ";
            }
            std::cout << std::endl;

            for (const auto &ef_recall : ef_recall_estimators)
            {
                std::cout << "Score: " << ef_recall.first << std::endl;
                for (const auto &pair : ef_recall.second)
                {
                    std::cout << "ef: " << pair.first << ", recall: " << pair.second << std::endl;
                }
            }
        }
    };
}